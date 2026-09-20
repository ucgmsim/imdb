#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"  # fiona (via source-modelling) has no 3.14 wheel yet
# dependencies = [
#     "numpy>=2",
#     "pandas>=3",
#     "h5py>=3.11",
#     "shapely",
#     "source-modelling",
#     "ucgmsim-imdb>=2026.9.2",
# ]
#
# [tool.uv]
# python-preference = "only-managed"  # avoid a broken/non-standard system Python on PATH
# ///

"""Ingest Felipe's NZ sim-validation database into an IMDB.

Each `<event_id>/` folder under `data_dir` is one real historical NZ
earthquake (event_id is a GeoNet CMT-style id, e.g. `2012p001887`), a single
realisation (no `_R<n>` split), source/rupture metadata from `realisation.json`,
simulated IMs from `intensity_measures.h5`. Observed records for the same events
come from the NZ GMDB v4.3 flat tables (rotd50/geom/eas), keyed on `evid` which is
identical to `event_id`.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from imdb import IMDB, schema
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from source_modelling import moment
from source_modelling.sources import Fault

GMDB_TABLES = ("rotd50", "geom", "eas")

TABLE_ORDER = (
    "db_meta", "notes", "im_units", "periods", "frequencies",
    "events", "realisations", "sites", "site_event", "records",
    "psa_ims", "fas_ims", "scalars_ims",
)


def _decode(values: np.ndarray) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def event_ids(data_dir: Path) -> list[str]:
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def event_dir(data_dir: Path, event_id: str) -> Path:
    return data_dir / event_id


def h5_path(data_dir: Path, event_id: str) -> Path:
    return event_dir(data_dir, event_id) / "intensity_measures.h5"


def load_realisation(data_dir: Path, event_id: str) -> dict:
    return json.loads((event_dir(data_dir, event_id) / "realisation.json").read_text())


def fault_geometries(realisation: dict) -> dict[str, Fault]:
    return {
        name: Fault.from_corners(
            np.array([[c["latitude"], c["longitude"], c["depth"]] for c in entry["corners"]]).reshape(-1, 4, 3)
        )
        for name, entry in realisation["sources"]["source_geometries"].items()
    }


def initial_fault_name(causality_tree: dict[str, str | None]) -> str:
    return next(name for name, parent in causality_tree.items() if parent is None)


def source_wkt(geometries: dict[str, Fault]) -> str:
    planes = [plane for f in geometries.values() for plane in f.planes]
    return MultiPolygon([Polygon(p.corners[:, [1, 0]]) for p in planes]).wkt


def trace_wkt(geometries: dict[str, Fault]) -> str:
    planes = [plane for f in geometries.values() for plane in f.planes]
    return MultiLineString([LineString(p.corners[:2, [1, 0]]) for p in planes]).wkt


def domain_wkt(realisation: dict) -> str:
    corners = realisation["domain"]["domain"]
    points = [(c["longitude"], c["latitude"]) for c in corners]
    return Polygon(points).wkt


def build_event(event_id: str, realisation: dict) -> tuple[dict, dict[str, Fault]]:
    geometries = fault_geometries(realisation)
    magnitudes = realisation["magnitudes"]["magnitudes"]
    rakes = realisation["rakes"]["rakes"]
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]

    if len(geometries) == 1:
        (f,) = geometries.values()
        (seg_magnitude,) = magnitudes.values()
        dip, dip_dir = f.dip, f.dip_dir
        dtop, dbottom, length = f.top_m / 1000, f.bottom_m / 1000, f.length
        magnitude = moment.boldm_to_mw(seg_magnitude)
    else:
        # Cascading multi-fault rupture: no single dip/length/magnitude fits, so the
        # scalar columns stay null and the per-segment breakdown goes in metadata.
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
        "event_id": event_id,
        "magnitude": magnitude,
        "tect_type": "ACTIVE_SHALLOW",
        "dip": dip,
        "dip_dir": dip_dir,
        "dtop": dtop,
        "dbottom": dbottom,
        "length": length,
        "source_wkt": source_wkt(geometries),
        "trace_wkt": trace_wkt(geometries),
        "domain_wkt": domain_wkt(realisation),
        "metadata": json.dumps({"nshm_rupture_name": realisation["metadata"]["name"], "segments": segments}),
    }
    return event, geometries


def build_realisation(event_id: str, geometries: dict[str, Fault], realisation: dict) -> dict:
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
        "rel_id": event_id,
        "event_id": event_id,
        "magnitude": total_magnitude,
        "rake": rakes[initial_fault],
        "hypo_lat": hypo_lat,
        "hypo_lon": hypo_lon,
        "hypo_depth": hypo_depth_m / 1000,
        "metadata": json.dumps({"seeds": realisation["seeds"]}),
    }


def load_sim_sites(data_dir: Path, ids: list[str]) -> pd.DataFrame:
    """Sites from every event's simulation grid, deduped by station id.

    The grid is virtual (not real recording stations) and is regenerated per event, so
    the same station id can carry slightly different lat/lon between events (~80m mean,
    ~200m max jitter observed). Treated as the same physical site; first-seen lat/lon wins.
    """
    frames = []
    for event_id in ids:
        with h5py.File(h5_path(data_dir, event_id)) as h5:
            stations = _decode(h5["station"][:])
            lat = h5["latitude"][:]
            lon = h5["longitude"][:]
        frames.append(pd.DataFrame({"site_id": stations, "lat": lat, "lon": lon}))
    sites = pd.concat(frames, ignore_index=True).drop_duplicates("site_id", keep="first").reset_index(drop=True)
    sites["is_real"] = False
    return sites


def build_sim_site_event(event_id: str, geometries: dict[str, Fault], h5: h5py.File) -> pd.DataFrame:
    stations = _decode(h5["station"][:])
    df = pd.DataFrame(
        {
            "site_id": stations,
            "event_id": event_id,
            "rrup": h5["rrup"][:],
            "rjb": h5["rjb"][:],
        }
    )
    if len(geometries) == 1:
        (single_fault,) = geometries.values()
        latlon = np.column_stack([h5["latitude"][:], h5["longitude"][:]])
        rx, ry = single_fault.rx_ry_distance(latlon)
        df["rx"] = np.asarray(rx) / 1000
        df["ry"] = np.asarray(ry) / 1000
    return df


def build_sim_records(event_id: str, h5: h5py.File) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    stations = _decode(h5["station"][:])
    components = _decode(h5["component"][:])
    scalars = {im: h5[im][:] for im in schema.SCALAR_IMS if im in h5}
    pSA = h5["pSA"][:]  # (component, period, station)
    FAS = h5["FAS"][:]  # (component, station, frequency)

    n_components, n_stations = len(components), len(stations)
    records_df = pd.DataFrame(
        {
            "rel_id": event_id,
            "site_id": np.tile(stations, n_components),
            "component": np.repeat(components, n_stations),
            "kind": "simulated",
            **{im: values.reshape(-1) for im, values in scalars.items()},
        }
    )
    psa_arr = pSA.transpose(0, 2, 1).reshape(-1, pSA.shape[1]).astype(np.float32)
    fas_arr = FAS.reshape(-1, FAS.shape[2]).astype(np.float32)
    return records_df, psa_arr, fas_arr


def load_gmdb_table(path: Path, ids: set[str], chunksize: int = 500_000) -> pd.DataFrame:
    """Read one GMDB flat table, keeping only rows for `ids` (matched on `evid`)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    chunks = [
        chunk[chunk["evid"].astype(str).isin(ids)]
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False)
    ]
    matched = [c for c in chunks if len(c)]
    return pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=header)


def load_gmdb_tables(gmdb_dir: Path, ids: list[str]) -> dict[str, pd.DataFrame]:
    id_set = set(ids)
    return {name: load_gmdb_table(gmdb_dir / f"ground_motion_im_table_{name}_flat.csv", id_set) for name in GMDB_TABLES}


def build_observed_sites(gmdb: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for df in gmdb.values():
        if df.empty:
            continue
        frame = df[["sta", "sta_lat", "sta_lon"]].rename(
            columns={"sta": "site_id", "sta_lat": "lat", "sta_lon": "lon"}
        )
        if "Vs30" in df:
            frame = frame.assign(vs30=df["Vs30"], z1p0=df["Z1.0"] / 1000, z2p5=df["Z2.5"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["site_id", "lat", "lon", "vs30", "z1p0", "z2p5", "is_real"])
    all_sites = pd.concat(frames, ignore_index=True)

    rounded = all_sites.assign(
        lat_r=all_sites["lat"].round(6),
        lon_r=all_sites["lon"].round(6),
        vs30_r=all_sites["vs30"].round(4),
        z1p0_r=all_sites["z1p0"].round(4),
        z2p5_r=all_sites["z2p5"].round(4),
    )
    check_cols = ["lat_r", "lon_r", "vs30_r", "z1p0_r", "z2p5_r"]
    inconsistent = rounded.groupby("site_id")[check_cols].nunique().gt(1).any(axis=1)
    if inconsistent.any():
        raise ValueError(f"Observed site metadata disagrees across records for: {inconsistent[inconsistent].index.tolist()}")

    sites = all_sites.drop_duplicates("site_id").reset_index(drop=True)
    sites["is_real"] = True
    return sites


def build_observed_site_event(event_id: str, gmdb: dict[str, pd.DataFrame], sim_site_ids: set[str]) -> pd.DataFrame:
    """site_event rows for observed stations, excluding ones the sim grid already covers.

    Some observed stations coincide with a sim grid site (same site_id, same event, per
    `build_event`/`main`'s site-merge note); site_event's logical key is (site_id, event_id),
    so only one row can exist per pair. The sim-computed row wins for those.
    """
    df = gmdb["rotd50"]
    df = df[(df["evid"].astype(str) == event_id) & (~df["sta"].isin(sim_site_ids))]
    return pd.DataFrame(
        {
            "site_id": df["sta"],
            "event_id": event_id,
            "rrup": df["r_rup"],
            "rjb": df["r_jb"],
            "rx": df["r_x"],
            "ry": df["r_y"],
        }
    )


def _im_columns(header: list[str], prefix: str, grid: list[float]) -> list[str]:
    cols = [c for c in header if c.startswith(prefix)]
    if len(cols) != len(grid):
        raise ValueError(f"{prefix}: expected {len(grid)} columns, found {len(cols)}")
    return cols


def build_observed_records_rotd50(event_id: str, df: pd.DataFrame, periods: list[float]) -> tuple[pd.DataFrame, np.ndarray]:
    df = df[df["evid"].astype(str) == event_id]
    records_df = pd.DataFrame(
        {
            "rel_id": event_id,
            "site_id": df["sta"],
            "component": "rotd50",
            "kind": "observed",
            "PGA": df["PGA"],
            "PGV": df["PGV"],
        }
    )
    psa_cols = _im_columns(df.columns.tolist(), "pSA_", periods)
    psa_arr = df[psa_cols].to_numpy(dtype=np.float32)
    return records_df, psa_arr


def build_observed_records_geom(event_id: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["evid"].astype(str) == event_id]
    return pd.DataFrame(
        {
            "rel_id": event_id,
            "site_id": df["sta"],
            "component": "geom",
            "kind": "observed",
            "CAV": df["CAV"],
            "AI": df["AI"],
            "Ds575": df["Ds575"],
            "Ds595": df["Ds595"],
        }
    )


def build_observed_records_eas(event_id: str, df: pd.DataFrame, frequencies: list[float]) -> tuple[pd.DataFrame, np.ndarray]:
    df = df[df["evid"].astype(str) == event_id]
    records_df = pd.DataFrame(
        {"rel_id": event_id, "site_id": df["sta"], "component": "eas", "kind": "observed"}
    )
    fas_cols = _im_columns(df.columns.tolist(), "FAS_", frequencies)
    fas_arr = df[fas_cols].to_numpy(dtype=np.float32)
    return records_df, fas_arr


def flush_to_disk(db: IMDB, db_path: Path) -> None:
    """Copy the in-memory database out to `db_path`, in FK-safe table order.

    `COPY FROM DATABASE` doesn't respect foreign-key dependency order, so tables
    are created and copied one at a time instead.
    """
    con = db.con
    con.raw_sql(f"ATTACH '{db_path}' AS disk_db")
    con.raw_sql("USE disk_db")
    for statement in schema.DDL.strip().split(";"):
        if statement.strip():
            con.raw_sql(statement)
    con.raw_sql("USE memory")
    for table in TABLE_ORDER:
        con.raw_sql(f"INSERT INTO disk_db.{table} SELECT * FROM memory.{table}")
    con.raw_sql("DETACH disk_db")


def main(data_dir: Path, gmdb_dir: Path, db_path: Path) -> None:
    ids = event_ids(data_dir)

    sim_sites = load_sim_sites(data_dir, ids)
    gmdb = load_gmdb_tables(gmdb_dir, ids)
    observed_sites = build_observed_sites(gmdb)

    with h5py.File(h5_path(data_dir, ids[0])) as h5:
        periods = h5["period"][:].tolist()
        frequencies = h5["frequency"][:].tolist()

    db = IMDB.create(
        ":memory:",
        periods=periods,
        frequencies=frequencies,
        components=schema.COMPONENTS,
        db_meta={"dataset_id": "nz_sim_validation"},
    )
    # The simulation grid deliberately places a virtual site at each real observed
    # station's location (same site_id, e.g. "AMBC") for direct sim-vs-obs comparison.
    # Where a site_id is both, the observed row (real coordinates, vs30/z, is_real=True)
    # wins over the sim placeholder.
    all_sites = pd.concat([sim_sites, observed_sites], ignore_index=True).drop_duplicates(
        "site_id", keep="last"
    )
    db.add_sites(all_sites)

    for event_id in ids:
        realisation = load_realisation(data_dir, event_id)
        event, geometries = build_event(event_id, realisation)
        db.add_events(pd.DataFrame([event]))
        db.add_realisations(pd.DataFrame([build_realisation(event_id, geometries, realisation)]))

        with h5py.File(h5_path(data_dir, event_id)) as h5:
            assert h5["period"][:].tolist() == periods, f"{event_id}: period grid mismatch"
            assert h5["frequency"][:].tolist() == frequencies, f"{event_id}: frequency grid mismatch"
            sim_site_ids = set(_decode(h5["station"][:]))
            db.add_site_event(build_sim_site_event(event_id, geometries, h5))
            records_df, psa_arr, fas_arr = build_sim_records(event_id, h5)
        db.add_records(records_df, pSA=psa_arr, FAS=fas_arr)

        observed_site_event = build_observed_site_event(event_id, gmdb, sim_site_ids)
        if not observed_site_event.empty:
            db.add_site_event(observed_site_event)

        rotd50_records, rotd50_psa = build_observed_records_rotd50(event_id, gmdb["rotd50"], periods)
        if not rotd50_records.empty:
            db.add_records(rotd50_records, pSA=rotd50_psa)

        geom_records = build_observed_records_geom(event_id, gmdb["geom"])
        if not geom_records.empty:
            db.add_records(geom_records)

        eas_records, eas_fas = build_observed_records_eas(event_id, gmdb["eas"], frequencies)
        if not eas_records.empty:
            db.add_records(eas_records, FAS=eas_fas)

    problems = db.validate()
    print("validate():", problems or "clean")
    print("events:", len(db.get_events()))
    print("realisations:", len(db.get_realisations()))
    print("sites:", len(db.get_sites()))
    print("records by kind:")
    print(db.get_records()["kind"].value_counts())
    flush_to_disk(db, db_path)
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="2026-06-24_runs_v8/results/ directory, containing one folder per event_id")
    parser.add_argument("gmdb_dir", type=Path, help="NZ GMDB v4.3_final/Tables directory")
    parser.add_argument("db_path", type=Path, help="Output IMDB path")
    args = parser.parse_args()
    main(args.data_dir, args.gmdb_dir, args.db_path)
