#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2",
#     "pandas>=3",
#     "h5py",
#     "source-modelling",
#     "ucgmsim-imdb",
#     "oq-wrapper",
# ]
#
# [tool.uv.sources]
# ucgmsim-imdb = { git = "ssh://git@github.com/ucgmsim/imdb.git", branch = "alpine-imdb" }
# ///

"""Ingest the Alpine vs30 site-table-update rerun into an IMDB.

Each `<fault>_R<n>` folder under `data_dir` (fault in base/clarence/hope/wairau, n in
1-3) is one realisation of the `fault` event. Source/rupture metadata comes from each
folder's `realisation.json`; IM values come from `intensity_measures.site_table_vs30.h5`
only (the grid_vs30 variant is not ingested). Empirical NSHM2022 pSA GMM predictions are
computed per event from `oq_wrapper` and ingested alongside the simulated records.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import oq_wrapper as oqw
import pandas as pd
from imdb import IMDB, schema
from source_modelling import moment
from source_modelling.sources import Fault

FAULTS = ["base", "clarence", "hope", "wairau"]
REL_NUMBERS = [1, 2, 3]
SCALAR_IMS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]
DEFAULT_SITE_TABLE = Path("/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v4.3_final/Tables/site_table.csv")


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def rel_dir(data_dir: Path, fault: str, n: int) -> Path:
    return data_dir / f"{fault}_R{n}"


def h5_path(data_dir: Path, fault: str, n: int) -> Path:
    return rel_dir(data_dir, fault, n) / "intensity_measures.site_table_vs30.h5"


def load_realisation(data_dir: Path, fault: str, n: int) -> dict:
    return json.loads((rel_dir(data_dir, fault, n) / "realisation.json").read_text())


def load_sites(data_dir: Path, site_table_path: Path) -> pd.DataFrame:
    """Site table: lat/lon from the h5, vs30 from vs30_comparison.csv, built once globally.

    z1p0/z2p5 come from the NZ GMDB site table (not present anywhere in this dataset) —
    required for the empirical GMM logic tree, not just informational.
    """
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
            frame = pd.DataFrame({"site_id": stations, "lat": lat, "lon": lon}).merge(
                vs30_df, on="site_id", how="left"
            )
            missing_vs30 = frame.loc[frame["vs30"].isna(), "site_id"].tolist()
            if missing_vs30:
                raise ValueError(
                    f"{fault}_R{n}: stations missing from vs30_comparison.csv: {missing_vs30}"
                )
            frames.append(frame)
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

    # Z1.0 in the GMDB table is metres, Z2.5 is already km (verified: Z2.5 > Z1.0/1000 for every station).
    z_table = pd.read_csv(site_table_path, usecols=["sta", "Z1.0", "Z2.5"]).rename(columns={"sta": "site_id"})
    sites = sites.merge(z_table, on="site_id", how="left")
    missing = sites.loc[sites["Z1.0"].isna() | sites["Z2.5"].isna(), "site_id"].tolist()
    if missing:
        raise ValueError(f"Missing Z1.0/Z2.5 in {site_table_path} for stations: {missing}")
    sites["z1p0"] = sites["Z1.0"] / 1000
    sites["z2p5"] = sites["Z2.5"]
    return sites.drop(columns=["Z1.0", "Z2.5"])


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


def build_gmm_records(
    ref_fault: Fault,
    site_event_df: pd.DataFrame,
    sites: pd.DataFrame,
    realisations: list[dict],
    periods: list[float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """NSHM2022 empirical pSA logic tree, one call per event covering all its realisations.

    dip/ztor/zbot/rx/ry have no single well-defined value for a cascading multi-segment
    rupture, so they're approximated from the initiating segment (`ref_fault`) — the same
    segment already used for rake/hypocentre in `build_realisation`. rrup/rjb stay the
    physically-correct min-across-segments values already computed in `site_event_df`.
    """
    latlon = sites[["lat", "lon"]].to_numpy()
    rx, ry = ref_fault.rx_ry_distance(latlon)
    site_geom = pd.DataFrame(
        {
            "site_id": sites["site_id"],
            "vs30": sites["vs30"],
            "z1pt0": sites["z1p0"],
            "z2pt5": sites["z2p5"],
            "rx": np.asarray(rx) / 1000,
            "ry": np.asarray(ry) / 1000,
        }
    ).merge(site_event_df[["site_id", "rrup", "rjb"]], on="site_id")

    rupture_df = pd.concat(
        [
            site_geom.assign(rel_id=rel["rel_id"], mag=rel["magnitude"], rake=rel["rake"], hypo_depth=rel["hypo_depth"])
            for rel in realisations
        ],
        ignore_index=True,
    )
    rupture_df["dip"] = ref_fault.dip
    rupture_df["ztor"] = ref_fault.top_m / 1000
    rupture_df["zbot"] = ref_fault.bottom_m / 1000
    rupture_df["vs30measured"] = True
    rupture_df["backarc"] = False

    # GMMs in this logic tree only cover periods up to 10s; beyond that oq_wrapper would
    # extrapolate (and warn per call), which isn't a real prediction, so don't ask for it.
    in_range = [p for p in periods if p <= 10]
    result = oqw.run_gmm_logic_tree(
        oqw.constants.GMMLogicTree.NSHM2022,
        oqw.constants.TectType["ACTIVE_SHALLOW"],
        rupture_df,
        "pSA",
        periods=in_range,
    )
    gmm_df = pd.DataFrame(
        {
            "rel_id": rupture_df["rel_id"],
            "site_id": rupture_df["site_id"],
            "component": "rotd50",
            "kind": "gmm",
            "gmm_key": "NSHM2022",
        }
    )
    psa = np.full((len(rupture_df), len(periods)), np.nan, dtype=np.float32)
    psa_sigma = np.full((len(rupture_df), len(periods)), np.nan, dtype=np.float32)
    in_range_cols = [i for i, p in enumerate(periods) if p <= 10]
    psa[:, in_range_cols] = np.exp(result[[f"pSA_{p}_mean" for p in in_range]].to_numpy(dtype=np.float32))
    psa_sigma[:, in_range_cols] = result[[f"pSA_{p}_std_Total" for p in in_range]].to_numpy(dtype=np.float32)
    return gmm_df, psa, psa_sigma


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


def main(data_dir: Path, db_path: Path, site_table_path: Path) -> None:
    sites = load_sites(data_dir, site_table_path)

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
        causality_tree = event_realisation["rupture_propagation"]["rupture_causality_tree"]
        ref_fault = geometries[initial_fault_name(causality_tree)]
        db.add_events(pd.DataFrame([event]))
        site_event_df = build_site_event(fault, geometries, sites)
        db.add_site_event(site_event_df)

        realisations = []
        for n in REL_NUMBERS:
            with h5py.File(h5_path(data_dir, fault, n)) as h5:
                assert h5["period"][:].tolist() == periods, f"{fault}_R{n}: period grid mismatch"
                assert h5["frequency"][:].tolist() == frequencies, f"{fault}_R{n}: frequency grid mismatch"
            realisations.append(build_realisation(fault, n, geometries, load_realisation(data_dir, fault, n)))
        db.add_realisations(pd.DataFrame(realisations))

        gmm_df, psa, psa_sigma = build_gmm_records(ref_fault, site_event_df, sites, realisations, periods)
        db.add_records(gmm_df, pSA=psa, pSA_sigma=psa_sigma)

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
    parser.add_argument(
        "--site-table",
        type=Path,
        default=DEFAULT_SITE_TABLE,
        help="NZ GMDB site_table.csv, source of Z1.0/Z2.5 (not present in data_dir)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.db_path, args.site_table)
