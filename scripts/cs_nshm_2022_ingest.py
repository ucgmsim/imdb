#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"  # fiona (via source-modelling) has no 3.14 wheel yet
# dependencies = [
#     "numpy>=2",
#     "pandas>=3",
#     "h5py>=3.11",
#     "shapely",
#     "source-modelling>=2026.8.3",
#     "ucgmsim-imdb>=2026.9.2",
# ]
#
# [tool.uv]
# python-preference = "only-managed"  # avoid a broken/non-standard system Python on PATH
# ///

"""Ingest the cs_nshm_2022 run3 simulation campaign into an IMDB.

One event per NSHM 2022 crustal rupture and one realisation per simulated
`R<n>`. The event_id is the campaign's rupture id, which is the rupture's
crustal `nshm_id` in nshmdb, NOT its `rupture_id`; the rel_id is
`<rupture>_R<n>`.

- IMs, distances, vs30 and z1.0/z2.5 come from each realisation's
  `intensity_measures.h5`, as written by the workflow's im-calc: netCDF4, one
  group per IM, one dataset per component.
- Source and rupture metadata come from its `realisation.json`.
- The rupture's NSHM magnitude and annual rate come from a CSV exported from
  nshmdb, which is not available where this runs.
- Canonical site coordinates come from the campaign's `stations_input.ll`.

Trimmed for size: only the geom and rotd50 components, pSA on 25 of im-calc's
111 periods, and no FAS or empirical-GMM records.

Every magnitude is BoldM (Hanks & Kanamori 1979 eq. 7), the group convention.
nz_sim_validation_ingest.py converts to Mw instead.
"""

import argparse
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from imdb import IMDB, schema
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from source_modelling import moment
from source_modelling.sources import Fault

# 0.1 s steps to 1 s, then 1 s steps. im-calc computed no 16-19 s; 20 s is its last.
PSA_PERIODS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 20.0,
)  # fmt: skip
COMPONENTS = ("geom", "rotd50")
N_IM_CALC_PERIODS = 111
SITE_FIELDS = ("vs30", "z1pt0", "z2pt5")
DISTANCE_FIELDS = ("rrup", "rjb", "rx", "ry")
# How closely realisation.json must reproduce the h5's own attributes. Measured
# over all 221 realisations done on 2026-09-21: magnitude identical, hypocentre
# within 3e-14 degrees.
TOLERANCES = {"magnitude": 1e-9, "hypo_lat": 1e-9, "hypo_lon": 1e-9, "hypo_depth": 1e-6}

DB_META = {
    "dataset_id": "cs_nshm_2022_run3",
    "magnitude_convention": (
        "BoldM (Hanks & Kanamori 1979 eq. 7). events.magnitude is the NSHM 2022 "
        "catalogue magnitude; realisations.magnitude is the realisation's own "
        "moment-summed total."
    ),
    "psa_period_selection": (
        "25 of im-calc's 111 periods: 0.1-1.0 s by 0.1 s, 2-15 s by 1 s, and 20 s. "
        "im-calc computed no 16-19 s."
    ),
    "station_coordinates": (
        "sites.lat/lon are the canonical stations_input.ll coordinates. The "
        "simulation snapped each station to its rupture's grid (~0.1 km away, so "
        "up to ~0.25 km apart between ruptures); those coordinates are not stored."
    ),
}


# ---- inputs ------------------------------------------------------------------


def select_period_indices(
    available: np.ndarray, wanted: Sequence[float] = PSA_PERIODS
) -> np.ndarray:
    """Column of each wanted period in im-calc's period grid.

    Raises ValueError unless every wanted period matches exactly one available one.
    """
    indices = []
    for period in wanted:
        (matches,) = np.nonzero(np.isclose(available, period, rtol=1e-9, atol=0.0))
        if len(matches) != 1:
            raise ValueError(
                f"pSA period {period} s matches {len(matches)} of im-calc's "
                "periods, expected exactly 1"
            )
        indices.append(int(matches[0]))
    return np.array(indices)


def parse_realisation_ids(lines: Iterable[str]) -> list[tuple[str, int]]:
    """Parse `<rupture>/R<n>` lines into (rupture, n) pairs, in ingest order.

    Blank lines are skipped and duplicates dropped; the order is by integer
    rupture id, then realisation.
    """
    ids = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"(\d+)/R(\d+)", line)
        if match is None:
            raise ValueError(f"not a <rupture>/R<n> id: {line!r}")
        ids.add((str(int(match[1])), int(match[2])))
    return sorted(ids, key=lambda rel: (int(rel[0]), rel[1]))


def group_by_rupture(realisations: Sequence[tuple[str, int]]) -> dict[str, list[int]]:
    """Realisation numbers per rupture, keeping the input order."""
    grouped: dict[str, list[int]] = {}
    for rupture, n in realisations:
        grouped.setdefault(rupture, []).append(n)
    return grouped


def rel_id(rupture: str, n: int) -> str:
    """The realisation's stable id."""
    return f"{rupture}_R{n}"


def load_stations_input(path: Path) -> pd.DataFrame:
    """Canonical station coordinates (`lon lat name` per line), indexed by name."""
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=["lon", "lat", "site_id"], dtype={"site_id": str}
    )
    duplicated = df["site_id"][df["site_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"{path}: duplicate station names, e.g. {list(duplicated[:5])}")
    return df.set_index("site_id")


def load_nshm_attributes(path: Path) -> pd.DataFrame:
    """Rows of export_nshm_rupture_attributes.py's CSV, indexed by rupture id."""
    return pd.read_csv(path, dtype={"rupture_id": str}).set_index("rupture_id")


def is_real_station(site_id: str) -> bool:
    """GeoNet station codes are 3-4 characters; the virtual grid's codes are 7."""
    return len(site_id) != 7


# ---- intensity_measures.h5 ---------------------------------------------------


@dataclass
class RealisationIMs:
    """One realisation's intensity_measures.h5, reduced to what this database keeps."""

    stations: np.ndarray
    """Station names, str."""
    site: dict[str, np.ndarray]
    """SITE_FIELDS per station."""
    distances: dict[str, np.ndarray]
    """DISTANCE_FIELDS per station, km."""
    psa: dict[str, np.ndarray]
    """Component -> (n_stations, len(PSA_PERIODS)) float32."""
    scalars: dict[str, dict[str, np.ndarray]]
    """Scalar IM -> component -> (n_stations,) float32; rotd-undefined IMs omitted."""
    attrs: dict[str, float]
    """The file's magnitude and hypocentre attributes (TOLERANCES keys)."""


def _decode(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_ims(h5_path: Path) -> RealisationIMs:
    """Read the parts of one intensity_measures.h5 this database keeps."""
    with h5py.File(h5_path, "r") as h5:
        psa_group = h5["pSA"]
        periods = psa_group["period"][:]
        if len(periods) != N_IM_CALC_PERIODS:
            raise ValueError(
                f"{h5_path}: {len(periods)} pSA periods, expected {N_IM_CALC_PERIODS}"
            )
        columns = select_period_indices(periods)
        raw_stations = psa_group["station"][:]
        scalars: dict[str, dict[str, np.ndarray]] = {}
        for im in schema.SCALAR_IMS:
            if not np.array_equal(h5[im]["station"][:], raw_stations):
                raise ValueError(f"{h5_path}: {im} lists stations in a different order from pSA")
            scalars[im] = {}
            for component in COMPONENTS:
                if component in h5[im]:
                    scalars[im][component] = h5[im][component][:].astype(np.float32)
                elif not (component.startswith("rotd") and im in schema.ROTD_UNDEFINED):
                    raise ValueError(f"{h5_path}: {im} has no {component} component")
        return RealisationIMs(
            stations=_decode(raw_stations),
            site={key: psa_group[key][:].astype(np.float64) for key in SITE_FIELDS},
            distances={key: psa_group[key][:].astype(np.float64) for key in DISTANCE_FIELDS},
            psa={c: psa_group[c][:][:, columns].astype(np.float32) for c in COMPONENTS},
            scalars=scalars,
            attrs={key: float(np.atleast_1d(h5.attrs[key])[0]) for key in TOLERANCES},
        )


def check_finite(rel: str, ims: RealisationIMs) -> None:
    """Raise if any kept IM, site or distance value is NaN or infinite."""
    arrays = {
        **{f"pSA/{c}": values for c, values in ims.psa.items()},
        **{f"{im}/{c}": values for im, by_c in ims.scalars.items() for c, values in by_c.items()},
        **ims.site,
        **ims.distances,
    }
    for name, values in arrays.items():
        bad = ~np.isfinite(values)
        if bad.any():
            station = ims.stations[np.argwhere(bad)[0][0]]
            raise ValueError(
                f"{rel}: {name} has {int(bad.sum())} non-finite values, the first at {station}"
            )


def check_matches_first_realisation(
    rel: str, first_rel: str, ims: RealisationIMs, first: RealisationIMs
) -> None:
    """Raise unless stations, site fields and distances equal the rupture's first realisation's."""
    if not np.array_equal(ims.stations, first.stations):
        raise ValueError(f"{rel}: station list differs from {first_rel}")
    for mine, theirs in ((ims.site, first.site), (ims.distances, first.distances)):
        for key, values in mine.items():
            if not np.array_equal(values, theirs[key]):
                raise ValueError(
                    f"{rel}: {key} differs from {first_rel}; site and distance fields "
                    "must be identical across a rupture's realisations"
                )


def build_records(rel: str, ims: RealisationIMs) -> tuple[pd.DataFrame, np.ndarray]:
    """IMDB records (one per station and component) and their pSA rows."""
    n = len(ims.stations)
    frames = []
    for component in COMPONENTS:
        frame = {"rel_id": rel, "site_id": ims.stations, "component": component, "kind": "simulated"}
        for im, by_component in ims.scalars.items():
            frame[im] = by_component.get(component, np.full(n, np.nan, dtype=np.float32))
        frames.append(pd.DataFrame(frame))
    records = pd.concat(frames, ignore_index=True)
    return records, np.concatenate([ims.psa[c] for c in COMPONENTS])


def build_site_event(rupture: str, ims: RealisationIMs) -> pd.DataFrame:
    """site_event rows for one rupture."""
    return pd.DataFrame({"site_id": ims.stations, "event_id": rupture, **ims.distances})


class SiteRegistry:
    """The sites added so far, with the vs30/z1pt0/z2pt5 first seen for each."""

    def __init__(self, stations_input: pd.DataFrame) -> None:
        self.stations_input = stations_input
        self.seen: pd.DataFrame | None = None

    def new_sites(self, rupture: str, ims: RealisationIMs) -> pd.DataFrame:
        """Rows for stations not added yet; raise if a known one disagrees."""
        here = pd.DataFrame(ims.site, index=pd.Index(ims.stations, name="site_id"))
        if self.seen is None:
            new = here
        else:
            known = here.index.isin(self.seen.index)
            if known.any():
                earlier = self.seen.loc[here.index[known]]
                mismatch = (here[known] != earlier).any(axis=1)
                if mismatch.any():
                    site = mismatch[mismatch].index[0]
                    raise ValueError(
                        f"rupture {rupture}: {site} has {here.loc[site].to_dict()}, but an "
                        f"earlier rupture gave {earlier.loc[site].to_dict()}"
                    )
            new = here[~known]
        missing = new.index.difference(self.stations_input.index)
        if len(missing):
            raise ValueError(
                f"rupture {rupture}: {len(missing)} stations missing from "
                f"stations_input.ll, e.g. {list(missing[:5])}"
            )
        self.seen = new if self.seen is None else pd.concat([self.seen, new])
        coords = self.stations_input.loc[new.index]
        return pd.DataFrame(
            {
                "site_id": new.index.to_numpy(),
                "lat": coords["lat"].to_numpy(),
                "lon": coords["lon"].to_numpy(),
                "vs30": new["vs30"].to_numpy(),
                "z1p0": new["z1pt0"].to_numpy(),
                "z2p5": new["z2pt5"].to_numpy(),
                "is_real": [is_real_station(site) for site in new.index],
            }
        )


# ---- realisation.json (geometry helpers as in nz_sim_validation_ingest.py) ----


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
    return Polygon([(c["longitude"], c["latitude"]) for c in corners]).wkt


def total_magnitude(realisation: dict) -> float:
    """The realisation's total magnitude, BoldM: segment moments summed, converted back."""
    magnitudes = realisation["magnitudes"]["magnitudes"]
    total_moment = sum(moment.magnitude_to_moment(m, bold_m=True) for m in magnitudes.values())
    return float(moment.moment_to_magnitude(total_moment, bold_m=True))


def build_event(event_id: str, realisation: dict, nshm: pd.Series) -> tuple[dict, dict[str, Fault]]:
    """The events row for a rupture, from one of its realisations and its NSHM attributes.

    Geometry is identical across a rupture's realisations, but magnitude and
    causality tree are not, so those live on the realisation instead.
    """
    geometries = fault_geometries(realisation)
    rakes = realisation["rakes"]["rakes"]
    tect_type = realisation.get("empirical", {}).get("tect_type", "").upper()
    if tect_type not in schema.TECT_TYPES:
        raise ValueError(f"rupture {event_id}: tect_type {tect_type!r} is not one of {schema.TECT_TYPES}")

    if len(geometries) == 1:
        (f,) = geometries.values()
        dip, dip_dir = f.dip, f.dip_dir
        dtop, dbottom, length = f.top_m / 1000, f.bottom_m / 1000, f.length
    else:
        # Cascading multi-fault rupture: no single dip/length fits, so the scalar
        # columns stay null and the per-segment breakdown goes in metadata.
        dip = dip_dir = dtop = dbottom = length = None

    segments = {
        name: {
            "dip_deg": f.dip,
            "dip_dir_deg": f.dip_dir,
            "dtop_km": f.top_m / 1000,
            "dbottom_km": f.bottom_m / 1000,
            "length_km": f.length,
            "rake": rakes[name],
        }
        for name, f in geometries.items()
    }
    event = {
        "event_id": event_id,
        "magnitude": float(nshm["magnitude_boldm"]),
        "tect_type": tect_type,
        "dip": dip,
        "dip_dir": dip_dir,
        "dtop": dtop,
        "dbottom": dbottom,
        "length": length,
        "source_wkt": source_wkt(geometries),
        "trace_wkt": trace_wkt(geometries),
        "domain_wkt": domain_wkt(realisation),
        "metadata": json.dumps(
            {
                "nshm_rupture_name": realisation["metadata"]["name"],
                "nshmdb_rupture_id": int(nshm["nshmdb_rupture_id"]),
                "annual_rate": float(nshm["annual_rate"]),
                "nshm_area_km2": float(nshm["area_km2"]),
                "nshm_length_km": float(nshm["length_km"]),
                "segments": segments,
            }
        ),
    }
    return event, geometries


def build_realisation(rupture: str, n: int, geometries: dict[str, Fault], realisation: dict) -> dict:
    """The realisations row for `<rupture>/R<n>`."""
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    initial_fault = initial_fault_name(causality_tree)
    hypocentre_sd = realisation["rupture_propagation"]["hypocentre"]
    hypo_lat, hypo_lon, hypo_depth_m = geometries[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
        np.array([hypocentre_sd["s"], hypocentre_sd["d"]])
    )
    return {
        "rel_id": rel_id(rupture, n),
        "event_id": rupture,
        "magnitude": total_magnitude(realisation),
        "rake": realisation["rakes"]["rakes"][initial_fault],
        "hypo_lat": float(hypo_lat),
        "hypo_lon": float(hypo_lon),
        "hypo_depth": float(hypo_depth_m) / 1000,
        "metadata": json.dumps(
            {
                "seeds": realisation["seeds"],
                "segment_magnitudes_boldm": realisation["magnitudes"]["magnitudes"],
                "rupture_causality_tree": causality_tree,
                "duration_s": realisation["domain"]["duration"],
            }
        ),
    }


def check_against_h5(rel: str, realisation_row: dict, attrs: dict[str, float]) -> None:
    """Raise unless realisation.json reproduces the h5's magnitude and hypocentre."""
    for key, tolerance in TOLERANCES.items():
        ours, theirs = realisation_row[key], attrs[key]
        if not abs(ours - theirs) <= tolerance:
            raise ValueError(
                f"{rel}: {key} from realisation.json is {ours!r}, but "
                f"intensity_measures.h5 says {theirs!r}"
            )


# ---- build and verify ----------------------------------------------------------


def check_row_counts(db: IMDB, expected: dict[str, int]) -> None:
    """Raise unless every table has exactly the expected number of rows."""
    for table, n in expected.items():
        found = int(db.con.table(table).count().to_pandas())
        if found != n:
            raise RuntimeError(f"{table}: {found} rows, expected {n}")


def spot_check(
    db_path: Path, share_dir: Path, n_realisations: int = 25, per_realisation: int = 40, seed: int = 0
) -> int:
    """Re-read random records from their h5 and compare exactly; return how many."""
    rng = np.random.default_rng(seed)
    checked = 0
    with IMDB(db_path) as db:
        rels = sorted(db.get_realisations().index)
        for rel in rng.choice(rels, size=min(n_realisations, len(rels)), replace=False):
            rupture, n = rel.rsplit("_R", 1)
            records = db.get_records(rel_ids=[rel])
            sample = records.sample(n=min(per_realisation, len(records)), random_state=seed)
            ids = sample.index.tolist()
            psa = db.get_psa(record_int_ids=ids)
            scalars = db.get_scalars(record_int_ids=ids)
            ims = read_ims(share_dir / rupture / f"R{n}" / "intensity_measures.h5")
            row_of = {site: i for i, site in enumerate(ims.stations)}
            for record_int_id, record in sample.iterrows():
                i, component = row_of[record["site_id"]], record["component"]
                stored = psa.loc[record_int_id].to_numpy(dtype=np.float32)
                if not np.array_equal(stored, ims.psa[component][i]):
                    raise AssertionError(f"{rel} {record['site_id']} {component}: pSA differs from the h5")
                for im in schema.SCALAR_IMS:
                    value = scalars.loc[record_int_id, im]
                    expected = ims.scalars[im].get(component)
                    if expected is None:
                        if not pd.isna(value):
                            raise AssertionError(f"{rel} {record['site_id']} {component}: {im} should be NULL")
                    elif np.float32(value) != expected[i]:
                        raise AssertionError(f"{rel} {record['site_id']} {component}: {im} differs from the h5")
                checked += 1
    return checked


def build_database(
    share_dir: Path,
    events_dir: Path,
    stations_input: Path,
    nshm_csv: Path,
    realisations: Sequence[tuple[str, int]],
    out_path: Path,
    source: str,
    script_commit: str,
    memory_limit: str | None = None,
) -> dict[str, int]:
    """Build the database at out_path and return its table row counts.

    Writes `<out_path>.partial` and renames it only once validation, the row
    counts and the spot check have all passed. A failure leaves the .partial
    behind for inspection, and the next run refuses to start until it is removed.
    """
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    partial = out_path.with_name(out_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"{partial} exists: a previous run died; remove it first")

    stations = load_stations_input(stations_input)
    nshm = load_nshm_attributes(nshm_csv)
    by_rupture = group_by_rupture(realisations)
    missing = sorted(set(by_rupture) - set(nshm.index), key=int)
    if missing:
        raise ValueError(f"ruptures missing from {nshm_csv}: {missing}")
    (nshmdb_file,) = nshm["nshmdb_file"].unique()
    (nshmdb_sha256,) = nshm["nshmdb_sha256"].unique()

    db = IMDB.create(
        partial,
        periods=list(PSA_PERIODS),
        components=COMPONENTS,
        db_meta={
            **DB_META,
            "source": source,
            "n_realisations": str(len(realisations)),
            "nshmdb": f"{nshmdb_file} sha256 {nshmdb_sha256}",
            "ingest_script_commit": script_commit,
        },
    )
    counts = dict.fromkeys(("events", "realisations", "sites", "site_event", "records"), 0)
    started = time.monotonic()
    try:
        if memory_limit:
            db.con.raw_sql(f"SET memory_limit = '{memory_limit}'")
        sites = SiteRegistry(stations)
        for index, (rupture, ns) in enumerate(by_rupture.items(), start=1):
            loaded = []
            for n in ns:
                rel = rel_id(rupture, n)
                realisation = json.loads((events_dir / rupture / f"R{n}" / "realisation.json").read_text())
                h5_path = share_dir / rupture / f"R{n}" / "intensity_measures.h5"
                if not h5_path.exists():
                    raise FileNotFoundError(f"{rel}: no {h5_path}")
                ims = read_ims(h5_path)
                check_finite(rel, ims)
                loaded.append((n, realisation, ims))

            first_n, first_realisation, first_ims = loaded[0]
            event, geometries = build_event(rupture, first_realisation, nshm.loc[rupture])
            rows = []
            for n, realisation, ims in loaded:
                row = build_realisation(rupture, n, geometries, realisation)
                check_against_h5(row["rel_id"], row, ims.attrs)
                check_matches_first_realisation(row["rel_id"], rel_id(rupture, first_n), ims, first_ims)
                rows.append(row)
            new_sites = sites.new_sites(rupture, first_ims)

            db.add_events(pd.DataFrame([event]))
            db.add_realisations(pd.DataFrame(rows))
            if len(new_sites):
                db.add_sites(new_sites)
            db.add_site_event(build_site_event(rupture, first_ims))
            for n, _, ims in loaded:
                records, psa = build_records(rel_id(rupture, n), ims)
                db.add_records(records, pSA=psa)
                counts["records"] += len(records)

            counts["events"] += 1
            counts["realisations"] += len(loaded)
            counts["sites"] += len(new_sites)
            counts["site_event"] += len(first_ims.stations)
            if index % 25 == 0 or index == len(by_rupture):
                print(
                    f"{index}/{len(by_rupture)} ruptures, {counts['records']} records, "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )

        problems = db.validate()
        if problems:
            raise RuntimeError(f"validate(): {problems}")
        expected = {**counts, "psa_ims": counts["records"], "scalars_ims": counts["records"]}
        check_row_counts(db, expected)
    finally:
        db.close()

    checked = spot_check(partial, share_dir)
    print(f"validate() clean; row counts match; spot check: {checked} records identical to their h5")
    partial.rename(out_path)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("share_dir", type=Path, help="run3 share/: <rupture>/R<n>/intensity_measures.h5")
    parser.add_argument("events_dir", type=Path, help="run3 events/: <rupture>/R<n>/realisation.json")
    parser.add_argument("stations_input", type=Path, help="the campaign's stations_input.ll (lon lat name)")
    parser.add_argument("nshm_csv", type=Path, help="campaign_nshm_ruptures.csv from export_nshm_rupture_attributes.py")
    parser.add_argument("realisations_file", type=Path, help="one <rupture>/R<n> per line")
    parser.add_argument("out", type=Path, help="output .duckdb; must not exist")
    parser.add_argument("--source", required=True, help="provenance for db_meta.source")
    parser.add_argument("--script-commit", required=True, help="git commit of this script, for db_meta")
    parser.add_argument(
        "--memory-limit",
        help="DuckDB memory_limit, e.g. 12GB. Without it DuckDB sizes itself from "
        "the node's RAM, not the job's memory allocation.",
    )
    args = parser.parse_args()

    realisations = parse_realisation_ids(args.realisations_file.read_text().splitlines())
    print(f"ingesting {len(realisations)} realisations into {args.out}", flush=True)
    counts = build_database(
        args.share_dir,
        args.events_dir,
        args.stations_input,
        args.nshm_csv,
        realisations,
        args.out,
        source=args.source,
        script_commit=args.script_commit,
        memory_limit=args.memory_limit,
    )
    print("row counts:", counts)


if __name__ == "__main__":
    main()
