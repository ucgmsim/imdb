#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2",
#     "pandas>=3",
#     "h5py",
#     "source-modelling",
#     "ucgmsim-imdb",
# ]
#
# [tool.uv.sources]
# ucgmsim-imdb = { git = "ssh://git@github.com/ucgmsim/imdb.git", branch = "alpine-imdb" }
# ///

"""Ingest the Alpine vs30 site-table-update rerun into an IMDB.

Each `<fault>_R<n>` folder under `data_dir` (fault in base/clarence/hope/wairau, n in
1-3) is one realisation of the `fault` event. Source/rupture metadata comes from each
folder's `realisation.json`; IM values come from `intensity_measures.site_table_vs30.h5`
only (the grid_vs30 variant is not ingested).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from imdb import IMDB, schema
from source_modelling import moment
from source_modelling.sources import Fault

FAULTS = ["base", "clarence", "hope", "wairau"]
REL_NUMBERS = [1, 2, 3]
SCALAR_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def rel_dir(data_dir: Path, fault: str, n: int) -> Path:
    return data_dir / f"{fault}_R{n}"


def h5_path(data_dir: Path, fault: str, n: int) -> Path:
    return rel_dir(data_dir, fault, n) / "intensity_measures.site_table_vs30.h5"


def load_realisation(data_dir: Path, fault: str, n: int) -> dict:
    return json.loads((rel_dir(data_dir, fault, n) / "realisation.json").read_text())


def load_sites(data_dir: Path) -> pd.DataFrame:
    """Site table: lat/lon from the h5, vs30 from vs30_comparison.csv, built once globally."""
    frames = []
    for fault in FAULTS:
        for n in REL_NUMBERS:
            with h5py.File(h5_path(data_dir, fault, n)) as h5:
                stations = _decode(h5["station"][:])
                lat = h5["latitude"][:]
                lon = h5["longitude"][:]
            vs30_df = pd.read_csv(
                rel_dir(data_dir, fault, n) / "vs30_comparison.csv",
                usecols=["station", "vs30_site_table"],
            ).rename(columns={"station": "site_id", "vs30_site_table": "vs30"})
            frames.append(pd.DataFrame({"site_id": stations, "lat": lat, "lon": lon}).merge(vs30_df, on="site_id"))
    all_sites = pd.concat(frames, ignore_index=True)

    rounded = all_sites.assign(
        lat_r=all_sites["lat"].round(6),
        lon_r=all_sites["lon"].round(6),
        vs30_r=all_sites["vs30"].round(4),
    )
    inconsistent = rounded.groupby("site_id")[["lat_r", "lon_r", "vs30_r"]].nunique().gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError(
            f"Site metadata disagrees across realisations for: {inconsistent[inconsistent].index.tolist()}"
        )

    sites = all_sites.drop_duplicates("site_id").reset_index(drop=True)
    sites["is_real"] = True
    return sites


def fault_geometries(realisation: dict) -> dict[str, Fault]:
    """One `Fault` per named source segment, built straight from realisation.json's corners."""
    return {
        name: Fault.from_corners(
            np.array([[c["latitude"], c["longitude"], c["depth"]] for c in entry["corners"]]).reshape(-1, 4, 3)
        )
        for name, entry in realisation["sources"]["source_geometries"].items()
    }


def initial_fault_name(causality_tree: dict[str, str | None]) -> str:
    return next(name for name, parent in causality_tree.items() if parent is None)


def build_event(fault: str, realisation: dict) -> tuple[dict, dict[str, Fault]]:
    geometries = fault_geometries(realisation)
    magnitudes = realisation["magnitudes"]["magnitudes"]
    rakes = realisation["rakes"]["rakes"]
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]

    if len(geometries) == 1:
        # Single-segment fault: the event's scalar geometry/magnitude are well-defined.
        (f,) = geometries.values()
        (seg_magnitude,) = magnitudes.values()
        dip, dip_dir = f.dip, f.dip_dir
        dtop, dbottom, length = f.top_m / 1000, f.bottom_m / 1000, f.length
        magnitude = moment.boldm_to_mw(seg_magnitude)
    else:
        # Cascading multi-fault rupture: no single dip/length/magnitude fits a
        # 12-20 segment rupture, so the scalar columns stay null and the per-segment
        # breakdown goes in metadata instead.
        dip = dip_dir = dtop = dbottom = length = magnitude = None

    segments = {
        name: {
            "magnitude_mw": moment.boldm_to_mw(magnitudes[name]),
            "rake": rakes[name],
            "dip_deg": f.dip,
            "dip_dir_deg": f.dip_dir,
            "dtop_km": f.top_m / 1000,
            "dbottom_km": f.bottom_m / 1000,
            "length_km": f.length,
            "parent": causality_tree[name],
        }
        for name, f in geometries.items()
    }
    event = {
        "event_id": fault,
        "magnitude": magnitude,
        "tect_type": "ACTIVE_SHALLOW",
        "dip": dip,
        "dip_dir": dip_dir,
        "dtop": dtop,
        "dbottom": dbottom,
        "length": length,
        "metadata": json.dumps({"nshm_rupture_name": realisation["metadata"]["name"], "segments": segments}),
    }
    return event, geometries


def build_site_event(fault: str, geometries: dict[str, Fault], sites: pd.DataFrame) -> pd.DataFrame:
    latlon = sites[["lat", "lon"]].to_numpy()
    latlondepth = np.column_stack([latlon, np.zeros(len(sites))])
    # Closest distance to any segment of the rupture (physically correct for a cascade).
    rrup = np.min([np.atleast_1d(f.rrup_distance(latlondepth)) for f in geometries.values()], axis=0)
    rjb = np.min([np.atleast_1d(f.rjb_distance(latlondepth)) for f in geometries.values()], axis=0)

    df = pd.DataFrame(
        {"site_id": sites["site_id"], "event_id": fault, "rrup": rrup / 1000, "rjb": rjb / 1000}
    )
    if len(geometries) == 1:
        # rx/ry need a single reference trace, which only exists for a single-segment fault.
        (single_fault,) = geometries.values()
        rx, ry = single_fault.rx_ry_distance(latlon)
        df["rx"] = np.asarray(rx) / 1000
        df["ry"] = np.asarray(ry) / 1000
    return df


def build_realisation(fault: str, n: int, geometries: dict[str, Fault], realisation: dict) -> dict:
    magnitudes = realisation["magnitudes"]["magnitudes"]
    rakes = realisation["rakes"]["rakes"]
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    initial_fault = initial_fault_name(causality_tree)
    hypocentre_sd = realisation["rupture_propagation"]["hypocentre"]
    hypocentre = np.array([hypocentre_sd["s"], hypocentre_sd["d"]])

    total_moment = sum(moment.magnitude_to_moment(m, bold_m=True) for m in magnitudes.values())
    total_magnitude = moment.boldm_to_mw(moment.moment_to_magnitude(total_moment, bold_m=True))
    hypo_lat, hypo_lon, hypo_depth_m = geometries[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
        hypocentre
    )

    return {
        "rel_id": f"{fault}_R{n}",
        "event_id": fault,
        "magnitude": total_magnitude,
        "rake": rakes[initial_fault],
        "hypo_lat": hypo_lat,
        "hypo_lon": hypo_lon,
        "hypo_depth": hypo_depth_m / 1000,
        "metadata": json.dumps({"seeds": realisation["seeds"]}),
    }


def build_records(fault: str, n: int, data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with h5py.File(h5_path(data_dir, fault, n)) as h5:
        stations = _decode(h5["station"][:])
        components = _decode(h5["component"][:])
        scalars = {im: h5[im][:] for im in SCALAR_IMS}
        pSA = h5["pSA"][:]  # (component, period, station)
        FAS = h5["FAS"][:]  # (component, station, frequency)

    n_components, n_stations = len(components), len(stations)
    records_df = pd.DataFrame(
        {
            "rel_id": f"{fault}_R{n}",
            "site_id": np.tile(stations, n_components),
            "component": np.repeat(components, n_stations),
            "kind": "simulated",
            **{im: values.reshape(-1) for im, values in scalars.items()},
        }
    )
    psa_arr = pSA.transpose(0, 2, 1).reshape(-1, pSA.shape[1]).astype(np.float32)
    fas_arr = FAS.reshape(-1, FAS.shape[2]).astype(np.float32)
    return records_df, psa_arr, fas_arr


def main(data_dir: Path, db_path: Path) -> None:
    sites = load_sites(data_dir)

    with h5py.File(h5_path(data_dir, FAULTS[0], 1)) as h5:
        periods = h5["period"][:].tolist()
        frequencies = h5["frequency"][:].tolist()

    db = IMDB.create(
        db_path,
        periods=periods,
        frequencies=frequencies,
        components=schema.COMPONENTS,
        db_meta={"dataset_id": "alpine_vs30_site_table_update"},
    )
    db.add_sites(sites)

    for fault in FAULTS:
        event_realisation = load_realisation(data_dir, fault, 1)
        event, geometries = build_event(fault, event_realisation)
        db.add_events(pd.DataFrame([event]))
        db.add_site_event(build_site_event(fault, geometries, sites))

        realisations = []
        for n in REL_NUMBERS:
            with h5py.File(h5_path(data_dir, fault, n)) as h5:
                assert h5["period"][:].tolist() == periods, f"{fault}_R{n}: period grid mismatch"
                assert h5["frequency"][:].tolist() == frequencies, f"{fault}_R{n}: frequency grid mismatch"
            realisations.append(build_realisation(fault, n, geometries, load_realisation(data_dir, fault, n)))
        db.add_realisations(pd.DataFrame(realisations))

        for n in REL_NUMBERS:
            records_df, psa_arr, fas_arr = build_records(fault, n, data_dir)
            db.add_records(records_df, pSA=psa_arr, FAS=fas_arr)

    problems = db.validate()
    print("validate():", problems or "clean")
    print("events:", len(db.get_events()))
    print("realisations:", len(db.get_realisations()))
    print("sites:", len(db.get_sites()))
    print("records by kind:")
    print(db.get_records()["kind"].value_counts())
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="vs30_site_table_update/ directory")
    parser.add_argument("db_path", type=Path, help="Output IMDB path")
    args = parser.parse_args()
    main(args.data_dir, args.db_path)
