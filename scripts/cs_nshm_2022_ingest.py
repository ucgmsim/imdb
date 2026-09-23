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

"""Ingest the cs_nshm_2022 simulations into one IMDB.

Two simulation campaigns become one dataset. Each event is one NSHM 2022
crustal rupture and each realisation is one simulation of it:

- the main campaign: up to two realisations per rupture, each with its own
  randomly drawn magnitude, hypocentre, rupture propagation and slip;
- the pilot: one realisation for each of 293 ruptures. Its magnitude is the
  scaling relation's central value, and its fault geometry comes from an
  earlier NSHM fault database release.

A manifest CSV (cs_nshm_2022's scripts/build_imdb_manifest.py) lists every
realisation with its rel_id, pilot flag and file paths, so this script knows
nothing about either campaign's layout. Each event's realisations are R1..Rn
with no gaps and the pilot's last. The event_id is the rupture's crustal
`nshm_id` in nshmdb, NOT its `rupture_id`.

- IMs, distances, vs30 and z1.0/z2.5 come from each realisation's
  `intensity_measures.h5`, as written by the workflow's im-calc: netCDF4, one
  group per IM, one dataset per component.
- Source and rupture metadata come from its `realisation.json`.
- The rupture's NSHM magnitude and annual rate come from a CSV exported from
  nshmdb, which is not available where this runs.
- Canonical site coordinates come from the campaign's `stations_input.ll`.

An event's geometry, and the distances stored for it, come from its first main
realisation, or from the pilot where the main campaign has none. At a site only
the pilot covers, site_event holds the pilot's distances, flagged in its metadata.

Trimmed for size by default: only the geom and rotd50 components, pSA on 25 of
im-calc's 111 periods, and no FAS. --components, --psa-periods all and --fas
put the rest in. Empirical-GMM records are never included.

Every magnitude is BoldM (Hanks & Kanamori 1979 eq. 7), the group convention.
nz_sim_validation_ingest.py converts to Mw instead.
"""

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from imdb import IMDB, schema
from shapely import unary_union
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from source_modelling import moment
from source_modelling.sources import Fault

# 0.1 s steps to 1 s, then 1 s steps. im-calc computed no 16-19 s; 20 s is its last.
PSA_PERIODS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 20.0,
)  # fmt: skip
DEFAULT_COMPONENTS = ("geom", "rotd50")
# Components im-calc writes for pSA, PGA, PGV and PGD, and for FAS.
PSA_COMPONENTS = ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")
FAS_COMPONENTS = ("000", "090", "ver", "geom", "eas")
N_IM_CALC_PERIODS = 111
SITE_FIELDS = ("vs30", "z1pt0", "z2pt5")
DISTANCE_FIELDS = ("rrup", "rjb", "rx", "ry")
# How closely realisation.json must reproduce the h5's own attributes. Measured
# over all 221 realisations done on 2026-09-21: magnitude identical, hypocentre
# within 3e-14 degrees.
TOLERANCES = {"magnitude": 1e-9, "hypo_lat": 1e-9, "hypo_lon": 1e-9, "hypo_depth": 1e-6}
MANIFEST_FIELDS = (
    "rel_id", "event_id", "realisation", "pilot", "campaign", "original_id", "im_path", "realisation_path",
)  # fmt: skip
# site_event.metadata on a row whose distances come from the pilot's geometry
# although the event's geometry is the main campaign's.
PILOT_DISTANCES = json.dumps({"fault_geometry": "pilot"})
# The NSHM fault database releases behind each campaign's geometry.
MAIN_RELEASE = "v2026.08.3"
MAIN_NSHMDB_SHA256 = "3fe692cb9b22c769b6a9baa6a526b4eed989bd61ec054a344f0e74d82855bc4e"
PILOT_RELEASE = "pre-v2026.08"
PILOT_NSHMDB_SHA256 = "00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57"
N_TOP_PEAKS = 20

DB_META = {
    "dataset_id": "cs_nshm_2022",
    "description": (
        "Physics-based ground-motion simulations of New Zealand NSHM 2022 crustal "
        "ruptures. Each event is one rupture and each realisation is one simulation of "
        "it, with its own hypocentre, rupture propagation and slip. Most realisations "
        "come from the main campaign, up to two per rupture, which also draws each "
        "realisation's magnitude at random. The rest come from an earlier pilot "
        "campaign, are marked realisations.metadata.pilot = true, and use the scaling "
        "relation's central magnitude instead."
    ),
    "source": "cs_nshm_2022 physics-based simulations; see description.",
    "realisation_numbering": (
        "Each event's realisations are R1..Rn with no gaps. Where an event has a pilot "
        "realisation, it is always the last. The seeds in realisations.metadata identify "
        "each simulation uniquely."
    ),
    "pilot_realisations": (
        "realisations.metadata.pilot = true marks the pilot's realisations, at most one "
        "per event. Their magnitude is the magnitude-area scaling relation's central value "
        "rather than a random draw, so leave them out of between-realisation variability "
        "estimates. Their fault geometry comes from an earlier release of the NSHM fault "
        "database: the same trace and depths, with the bottom edge placed differently (by "
        "a median of about 210 m, at most 2.6 km)."
    ),
    "distances": (
        "site_event rrup, rjb, rx and ry are computed from the event's own fault geometry, "
        "events.source_wkt (the main campaign's where the event has a main-campaign "
        "realisation, else the pilot's), and have NULL metadata. One exception: in an "
        "event that also has main-campaign realisations, at a site only its pilot "
        "realisation covers, the distances come from the pilot's own geometry, which is "
        f"not events.source_wkt, and site_event.metadata is {PILOT_DISTANCES}. Epicentral "
        "and hypocentral distances are not stored; derive them from realisations.hypo_* "
        "and the site coordinates."
    ),
    "nshm_fault_database": (
        "events.metadata.fault_geometry_release names the NSHM2022DB release the event's "
        f"geometry comes from: {MAIN_RELEASE} (sha256 {MAIN_NSHMDB_SHA256}), or "
        f"{PILOT_RELEASE}, the earlier release the pilot used (sha256 "
        f"{PILOT_NSHMDB_SHA256}). events.magnitude and the NSHM attributes in "
        f"events.metadata come from {MAIN_RELEASE} for every event."
    ),
    "magnitude_convention": (
        "BoldM (Hanks & Kanamori 1979 eq. 7). events.magnitude is the NSHM 2022 "
        "catalogue magnitude; realisations.magnitude is the realisation's own "
        "moment-summed total."
    ),
    "station_coordinates": (
        "sites.lat/lon are each station's canonical location. Each simulation moved "
        "its stations onto its own computational grid (~0.1 km away, so up to ~0.25 km "
        "apart between simulations) and computed the distances there; those "
        "coordinates are not stored."
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


@dataclass(frozen=True)
class Content:
    """Which components, pSA periods and spectra go into the database."""

    components: tuple[str, ...] = DEFAULT_COMPONENTS
    all_periods: bool = False
    """All 111 of im-calc's pSA periods, instead of the 25 in PSA_PERIODS."""
    fas: bool = False
    """FAS, on im-calc's full frequency grid, for the components that have it."""

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("no components chosen")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"components listed twice: {self.components}")
        unknown = set(self.components) - set(PSA_COMPONENTS) - set(FAS_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown components {sorted(unknown)}; choose from {schema.COMPONENTS}")
        if "eas" in self.components and not self.fas:
            raise ValueError("eas has only FAS, so it needs --fas")
        if self.fas and not self.fas_components:
            raise ValueError(f"--fas needs a component that has FAS: one of {', '.join(FAS_COMPONENTS)}")

    @property
    def psa_components(self) -> tuple[str, ...]:
        """The chosen components that have pSA and the scalar IMs: all but eas."""
        return tuple(c for c in self.components if c in PSA_COMPONENTS)

    @property
    def fas_components(self) -> tuple[str, ...]:
        """The chosen components that get FAS: none without --fas."""
        return tuple(c for c in self.components if self.fas and c in FAS_COMPONENTS)


@dataclass(frozen=True)
class Grids:
    """The pSA periods and FAS frequencies the database is created with."""

    periods: tuple[float, ...]
    frequencies: tuple[float, ...]


DEFAULT_CONTENT = Content()
DEFAULT_GRIDS = Grids(periods=PSA_PERIODS, frequencies=())


def read_grids(h5_path: Path, content: Content) -> Grids:
    """The database's grids, taken from one realisation's h5 where the content needs its full grid."""
    with h5py.File(h5_path, "r") as h5:
        if not content.psa_components:
            periods: tuple[float, ...] = ()
        elif content.all_periods:
            periods = tuple(float(p) for p in h5["pSA"]["period"][:])
        else:
            periods = PSA_PERIODS
        frequencies = tuple(float(f) for f in h5["FAS"]["frequency"][:]) if content.fas else ()
    return Grids(periods=periods, frequencies=frequencies)


def content_meta(content: Content, grids: Grids) -> dict[str, str]:
    """db_meta entries describing what the options put in."""
    if not content.psa_components:
        periods = "not included"
    elif content.all_periods:
        periods = f"all {len(grids.periods)} of im-calc's periods, {min(grids.periods):g}-{max(grids.periods):g} s."
    else:
        periods = (
            "25 of im-calc's 111 periods: 0.1-1.0 s by 0.1 s, 2-15 s by 1 s, and 20 s. "
            "im-calc computed no 16-19 s."
        )
    if content.fas_components:
        fas = (
            f"all {len(grids.frequencies)} of im-calc's FAS frequencies, "
            f"{min(grids.frequencies):g}-{max(grids.frequencies):g} Hz, for "
            f"{', '.join(content.fas_components)}. rotd components have no FAS."
        )
    else:
        fas = "not included"
    return {"psa_period_selection": periods, "fas": fas}


def rel_id(event_id: str, n: int) -> str:
    """The realisation's stable id."""
    return f"{event_id}_R{n}"


@dataclass(frozen=True)
class ManifestRow:
    """One realisation to ingest, as listed in the build manifest."""

    rel_id: str
    event_id: str
    realisation: int
    pilot: bool
    campaign: str
    """Which simulation campaign; for checks only, never stored."""
    original_id: str
    """Where the result came from; for messages only, never stored."""
    im_path: Path
    realisation_path: Path

    @property
    def label(self) -> str:
        return f"{self.rel_id} ({self.original_id})"


def read_manifest(path: Path) -> list[ManifestRow]:
    """The manifest's rows in ingest order: events by integer id, then realisations."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for raw in reader:
            if raw["pilot"] not in ("true", "false"):
                raise ValueError(f"{raw['rel_id']}: pilot must be true or false, not {raw['pilot']!r}")
            rows.append(
                ManifestRow(
                    rel_id=raw["rel_id"],
                    event_id=raw["event_id"],
                    realisation=int(raw["realisation"]),
                    pilot=raw["pilot"] == "true",
                    campaign=raw["campaign"],
                    original_id=raw["original_id"],
                    im_path=Path(raw["im_path"]),
                    realisation_path=Path(raw["realisation_path"]),
                )
            )
    check_manifest(rows)
    missing = [
        f"{row.label}: {path}" for row in rows for path in (row.im_path, row.realisation_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed files are missing, e.g. " + "; ".join(missing[:5]))
    return sorted(rows, key=lambda row: (int(row.event_id), row.realisation))


def check_manifest(rows: Sequence[ManifestRow]) -> None:
    """Raise unless ids are consistent and unique, and every event is R1..Rn with at most one pilot, last.

    The pilot flag must also follow the campaign: no campaign may supply both
    pilot and non-pilot rows.
    """
    if not rows:
        raise ValueError("the manifest lists no realisations")
    pilot_campaigns = {row.campaign for row in rows if row.pilot}
    main_campaigns = {row.campaign for row in rows if not row.pilot}
    if pilot_campaigns & main_campaigns:
        raise ValueError(
            f"campaign {sorted(pilot_campaigns & main_campaigns)} has both pilot and non-pilot rows"
        )
    seen: set[str] = set()
    by_event: dict[str, list[ManifestRow]] = {}
    for row in rows:
        if not row.event_id.isdigit() or str(int(row.event_id)) != row.event_id:
            raise ValueError(f"{row.rel_id}: event_id {row.event_id!r} is not an unpadded integer")
        if row.rel_id != rel_id(row.event_id, row.realisation):
            raise ValueError(f"{row.rel_id}: expected {rel_id(row.event_id, row.realisation)} from event_id and realisation")
        if row.rel_id in seen:
            raise ValueError(f"{row.rel_id}: listed twice")
        seen.add(row.rel_id)
        by_event.setdefault(row.event_id, []).append(row)
    for event_id, group in by_event.items():
        numbers = sorted(row.realisation for row in group)
        if numbers != list(range(1, len(group) + 1)):
            raise ValueError(f"event {event_id}: realisations {numbers}, expected 1..{len(group)}")
        pilots = [row for row in group if row.pilot]
        if len(pilots) > 1:
            raise ValueError(f"event {event_id}: {len(pilots)} pilot realisations, expected at most 1")
        if pilots and pilots[0].realisation != len(group):
            raise ValueError(f"event {event_id}: the pilot is R{pilots[0].realisation}, but it must be the last, R{len(group)}")


def group_by_event(rows: Sequence[ManifestRow]) -> dict[str, list[ManifestRow]]:
    """Rows per event, keeping the input order."""
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_id, []).append(row)
    return grouped


def md5_of(path: Path) -> str:
    """Hex md5 of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


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
    """Rows of export_nshm_rupture_attributes.py's CSV, indexed by rupture id.

    Raises unless they all come from the release DB_META documents.
    """
    nshm = pd.read_csv(path, dtype={"rupture_id": str}).set_index("rupture_id")
    if set(nshm["nshmdb_sha256"]) != {MAIN_NSHMDB_SHA256}:
        raise ValueError(
            f"{path}: NSHM attributes must all come from {MAIN_RELEASE} "
            f"(sha256 {MAIN_NSHMDB_SHA256}), found {sorted(set(nshm['nshmdb_sha256']))}"
        )
    return nshm


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
    """Component -> (n_stations, n_periods) float32, for Content.psa_components."""
    fas: dict[str, np.ndarray]
    """Component -> (n_stations, n_frequencies) float32, for Content.fas_components."""
    n_frequencies: int
    """Length of the database's FAS grid; 0 without FAS."""
    scalars: dict[str, dict[str, np.ndarray]]
    """Scalar IM -> component -> (n_stations,) float32; rotd-undefined IMs omitted."""
    attrs: dict[str, float]
    """The file's magnitude and hypocentre attributes (TOLERANCES keys)."""
    peaks: dict[str, tuple[float, str]]
    """PGA and PGV (geom) -> (largest value, its station), whatever the database keeps."""


def _decode(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def _peak(values: np.ndarray, stations: np.ndarray) -> tuple[float, str]:
    i = int(np.nanargmax(values))
    station = stations[i]
    return float(values[i]), station.decode() if isinstance(station, bytes) else str(station)


def read_ims(
    h5_path: Path, content: Content = DEFAULT_CONTENT, grids: Grids = DEFAULT_GRIDS
) -> RealisationIMs:
    """Read the parts of one intensity_measures.h5 this database keeps."""
    with h5py.File(h5_path, "r") as h5:
        psa_group = h5["pSA"]
        periods = psa_group["period"][:]
        if len(periods) != N_IM_CALC_PERIODS:
            raise ValueError(
                f"{h5_path}: {len(periods)} pSA periods, expected {N_IM_CALC_PERIODS}"
            )
        columns = select_period_indices(periods, grids.periods)
        raw_stations = psa_group["station"][:]
        scalars: dict[str, dict[str, np.ndarray]] = {}
        for im in schema.SCALAR_IMS:
            if not np.array_equal(h5[im]["station"][:], raw_stations):
                raise ValueError(f"{h5_path}: {im} lists stations in a different order from pSA")
            scalars[im] = {}
            for component in content.psa_components:
                if component in h5[im]:
                    scalars[im][component] = h5[im][component][:].astype(np.float32)
                elif not (component.startswith("rotd") and im in schema.ROTD_UNDEFINED):
                    raise ValueError(f"{h5_path}: {im} has no {component} component")
        fas: dict[str, np.ndarray] = {}
        if content.fas:
            fas_group = h5["FAS"]
            if not np.array_equal(fas_group["frequency"][:], np.array(grids.frequencies)):
                raise ValueError(f"{h5_path}: FAS frequency grid differs from the database's")
            if not np.array_equal(fas_group["station"][:], raw_stations):
                raise ValueError(f"{h5_path}: FAS lists stations in a different order from pSA")
            fas = {c: fas_group[c][:].astype(np.float32) for c in content.fas_components}
        return RealisationIMs(
            stations=_decode(raw_stations),
            site={key: psa_group[key][:].astype(np.float64) for key in SITE_FIELDS},
            distances={key: psa_group[key][:].astype(np.float64) for key in DISTANCE_FIELDS},
            psa={c: psa_group[c][:][:, columns].astype(np.float32) for c in content.psa_components},
            fas=fas,
            n_frequencies=len(grids.frequencies),
            scalars=scalars,
            attrs={key: float(np.atleast_1d(h5.attrs[key])[0]) for key in TOLERANCES},
            peaks={im: _peak(h5[im]["geom"][:], raw_stations) for im in ("PGA", "PGV")},
        )


def check_finite(rel: str, ims: RealisationIMs) -> None:
    """Raise if any kept IM, site or distance value is NaN or infinite."""
    arrays = {
        **{f"pSA/{c}": values for c, values in ims.psa.items()},
        **{f"FAS/{c}": values for c, values in ims.fas.items()},
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
    """Raise unless stations, site fields and distances equal the event's first main realisation's."""
    if not np.array_equal(ims.stations, first.stations):
        raise ValueError(f"{rel}: station list differs from {first_rel}")
    for mine, theirs in ((ims.site, first.site), (ims.distances, first.distances)):
        for key, values in mine.items():
            if not np.array_equal(values, theirs[key]):
                raise ValueError(
                    f"{rel}: {key} differs from {first_rel}; site and distance fields "
                    "must be identical across an event's main realisations"
                )


Batch = tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]


def build_records(rel: str, ims: RealisationIMs, content: Content = DEFAULT_CONTENT) -> list[Batch]:
    """IMDB records (one per station and component), as add_records batches of (records, pSA, FAS).

    `eas` has only FAS, so its records form a batch of their own with no pSA
    and no scalar columns, and add_records writes no psa_ims or scalars_ims
    rows for them. A rotd record's FAS row is all NaN, so it gets no fas_ims row.
    """
    n = len(ims.stations)
    frames = []
    for component in content.psa_components:
        frame = {"rel_id": rel, "site_id": ims.stations, "component": component, "kind": "simulated"}
        for im, by_component in ims.scalars.items():
            frame[im] = by_component.get(component, np.full(n, np.nan, dtype=np.float32))
        frames.append(pd.DataFrame(frame))
    batches: list[Batch] = []
    if frames:
        psa = np.concatenate([ims.psa[c] for c in content.psa_components])
        fas = None
        if content.fas:
            no_fas = np.full((n, ims.n_frequencies), np.nan, dtype=np.float32)
            fas = np.concatenate([ims.fas.get(c, no_fas) for c in content.psa_components])
        batches.append((pd.concat(frames, ignore_index=True), psa, fas))
    if "eas" in content.fas_components:
        eas = pd.DataFrame({"rel_id": rel, "site_id": ims.stations, "component": "eas", "kind": "simulated"})
        batches.append((eas, None, ims.fas["eas"]))
    return batches


def _site_event_rows(event_id: str, stations: np.ndarray, distances: dict[str, np.ndarray], metadata: str | None) -> pd.DataFrame:
    return pd.DataFrame({"site_id": stations, "event_id": event_id, **distances, "metadata": metadata})


def build_site_event(
    event_id: str, main: RealisationIMs | None, pilot: RealisationIMs | None
) -> tuple[pd.DataFrame, int]:
    """site_event rows for one event, and how many carry the pilot's distances.

    The main campaign's distances wherever its realisations cover the site: they
    all share one station list and one set of distances, so the first covers
    them all. The pilot's only where nothing else exists, flagged when the
    event's geometry is the main campaign's and unflagged when the event is the
    pilot's alone.
    """
    if main is None:
        if pilot is None:
            raise ValueError(f"event {event_id}: no realisations")
        return _site_event_rows(event_id, pilot.stations, pilot.distances, None), 0
    rows = _site_event_rows(event_id, main.stations, main.distances, None)
    if pilot is None:
        return rows, 0
    gap = ~pd.Index(pilot.stations).isin(main.stations)
    flagged = _site_event_rows(
        event_id,
        pilot.stations[gap],
        {key: values[gap] for key, values in pilot.distances.items()},
        PILOT_DISTANCES,
    )
    return pd.concat([rows, flagged], ignore_index=True), int(gap.sum())


class SiteRegistry:
    """The sites added so far, with the vs30/z1pt0/z2pt5 first seen for each."""

    def __init__(self, stations_input: pd.DataFrame) -> None:
        self.stations_input = stations_input
        self.seen: pd.DataFrame | None = None

    def new_sites(self, label: str, ims: RealisationIMs) -> pd.DataFrame:
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
                        f"{label}: {site} has {here.loc[site].to_dict()}, but an "
                        f"earlier realisation gave {earlier.loc[site].to_dict()}"
                    )
            new = here[~known]
        missing = new.index.difference(self.stations_input.index)
        if len(missing):
            raise ValueError(
                f"{label}: {len(missing)} stations missing from "
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


def domain_polygon(realisation: dict) -> Polygon:
    corners = realisation["domain"]["domain"]
    return Polygon([(c["longitude"], c["latitude"]) for c in corners])


def domain_wkt(realisations: Sequence[dict]) -> str:
    """The union of the realisations' simulation domains, so it contains every site with data."""
    return unary_union([domain_polygon(r) for r in realisations]).wkt


def total_magnitude(realisation: dict) -> float:
    """The realisation's total magnitude, BoldM: segment moments summed, converted back."""
    magnitudes = realisation["magnitudes"]["magnitudes"]
    total_moment = sum(moment.magnitude_to_moment(m, bold_m=True) for m in magnitudes.values())
    return float(moment.moment_to_magnitude(total_moment, bold_m=True))


def build_event(
    event_id: str, defining: dict, realisations: Sequence[dict], nshm: pd.Series, release: str
) -> dict:
    """The events row for a rupture.

    Geometry comes from `defining`, the realisation whose geometry the event's
    unflagged distances share; the domain is the union over `realisations`.
    Magnitude and causality tree differ between realisations, so those live on
    the realisation instead.
    """
    geometries = fault_geometries(defining)
    rakes = defining["rakes"]["rakes"]
    tect_type = defining.get("empirical", {}).get("tect_type", "").upper()
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
    return {
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
        "domain_wkt": domain_wkt(realisations),
        "metadata": json.dumps(
            {
                "nshmdb_rupture_id": int(nshm["nshmdb_rupture_id"]),
                "annual_rate": float(nshm["annual_rate"]),
                "nshm_area_km2": float(nshm["area_km2"]),
                "nshm_length_km": float(nshm["length_km"]),
                "fault_geometry_release": release,
                "segments": segments,
            }
        ),
    }


def build_realisation(row: ManifestRow, realisation: dict) -> dict:
    """The realisations row, from the realisation's own realisation.json and geometry."""
    geometries = fault_geometries(realisation)
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    initial_fault = initial_fault_name(causality_tree)
    hypocentre_sd = realisation["rupture_propagation"]["hypocentre"]
    hypo_lat, hypo_lon, hypo_depth_m = geometries[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
        np.array([hypocentre_sd["s"], hypocentre_sd["d"]])
    )
    return {
        "rel_id": row.rel_id,
        "event_id": row.event_id,
        "magnitude": total_magnitude(realisation),
        "rake": realisation["rakes"]["rakes"][initial_fault],
        "hypo_lat": float(hypo_lat),
        "hypo_lon": float(hypo_lon),
        "hypo_depth": float(hypo_depth_m) / 1000,
        "metadata": json.dumps(
            {
                "pilot": row.pilot,
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


def load_realisation(row: ManifestRow, content: Content, grids: Grids) -> tuple[dict, RealisationIMs]:
    """Read one realisation's realisation.json and h5, naming the realisation in any error.

    Raises unless the realisation.json describes the manifest row's event.
    """
    try:
        realisation = json.loads(row.realisation_path.read_text())
        name = realisation["metadata"]["name"]
        if name != f"Rupture {row.event_id}":
            raise ValueError(f"realisation.json describes {name!r}, not event {row.event_id}")
        return realisation, read_ims(row.im_path, content, grids)
    except (KeyError, OSError, ValueError) as error:
        raise type(error)(f"{row.label}: {error}") from error


@dataclass
class Loaded:
    """One realisation read from disk."""

    row: ManifestRow
    realisation: dict
    ims: RealisationIMs


def release_memtables(db: IMDB) -> None:
    """Drop the in-memory tables ibis registers for every insert.

    ibis (12.0) registers each inserted DataFrame with DuckDB as an Arrow table
    and releases them only at interpreter exit, so without this every row
    written stays in memory, outside DuckDB's memory_limit, for the whole build.
    """
    raw = db.con.con
    for (name,) in raw.execute(
        "SELECT view_name FROM duckdb_views() WHERE temporary AND view_name LIKE 'ibis_pandas_memtable_%'"
    ).fetchall():
        raw.unregister(name)


def check_row_counts(db: IMDB, expected: dict[str, int]) -> None:
    """Raise unless every table has exactly the expected number of rows."""
    for table, n in expected.items():
        found = int(db.con.table(table).count().to_pandas())
        if found != n:
            raise RuntimeError(f"{table}: {found} rows, expected {n}")


# Counts the flagged site_event rows three ways, from the tables alone: the rows
# flagged; the (site, event) pairs where only a pilot realisation has records in
# an event that also has main realisations; and the overlap of the two.
FLAG_QUERY = """
WITH rel AS (
    SELECT rel_int_id, event_int_id,
           coalesce(json_extract_string(metadata, '$.pilot') = 'true', false) AS pilot
    FROM realisations
), site_rel AS (
    SELECT DISTINCT records.site_int_id, records.event_int_id, rel.pilot
    FROM records JOIN rel USING (rel_int_id)
), pilot_only AS (
    SELECT site_int_id, event_int_id FROM site_rel
    GROUP BY site_int_id, event_int_id HAVING bool_and(pilot)
), shared AS (
    SELECT event_int_id FROM rel GROUP BY event_int_id HAVING bool_or(pilot) AND bool_or(NOT pilot)
)
SELECT
    (SELECT count(*) FROM site_event WHERE metadata IS NOT NULL),
    (SELECT count(*) FROM pilot_only WHERE event_int_id IN (SELECT event_int_id FROM shared)),
    (SELECT count(*) FROM site_event JOIN pilot_only USING (site_int_id, event_int_id)
     WHERE site_event.metadata IS NOT NULL
       AND site_event.event_int_id IN (SELECT event_int_id FROM shared))
"""


# site_event must hold exactly the (site, event) pairs that have records: the
# pairs in site_event but not in records, and those in records but not in site_event.
SITE_EVENT_QUERY = """
WITH pairs AS (SELECT DISTINCT site_int_id, event_int_id FROM records)
SELECT
    (SELECT count(*) FROM site_event ANTI JOIN pairs USING (site_int_id, event_int_id)),
    (SELECT count(*) FROM pairs ANTI JOIN site_event USING (site_int_id, event_int_id))
"""


def check_site_event_matches_records(db: IMDB) -> None:
    """Raise unless site_event has a row for exactly the (site, event) pairs that have records."""
    extra, missing = db.con.raw_sql(SITE_EVENT_QUERY).fetchone()
    if extra or missing:
        raise RuntimeError(f"site_event: {extra} rows with no records, {missing} record pairs with no row")


def check_flags(db: IMDB, n_flagged: int) -> None:
    """Raise unless the flagged site_event rows are exactly the pilot-only sites of shared events."""
    flagged, pilot_only, both = db.con.raw_sql(FLAG_QUERY).fetchone()
    if not flagged == pilot_only == both == n_flagged:
        raise RuntimeError(
            f"site_event flags: {flagged} flagged rows, {pilot_only} pilot-only sites in "
            f"shared events, {both} in both, {n_flagged} written"
        )


def spot_check(
    db_path: Path,
    manifest: dict[str, ManifestRow],
    content: Content = DEFAULT_CONTENT,
    grids: Grids = DEFAULT_GRIDS,
    n_realisations: int = 25,
    per_realisation: int = 40,
    min_pilot: int = 5,
    seed: int = 0,
) -> int:
    """Re-read random records from their h5 and compare exactly; return how many.

    At least `min_pilot` of the sampled realisations are pilot ones, where there are any.
    """
    rng = np.random.default_rng(seed)
    pilot_rels = sorted(rel for rel, row in manifest.items() if row.pilot)
    main_rels = sorted(rel for rel, row in manifest.items() if not row.pilot)
    n_pilot = min(len(pilot_rels), max(min_pilot, n_realisations - len(main_rels)))
    n_main = min(len(main_rels), n_realisations - n_pilot)
    chosen = [
        *rng.choice(pilot_rels, size=n_pilot, replace=False),
        *rng.choice(main_rels, size=n_main, replace=False),
    ]
    checked = 0
    with IMDB(db_path) as db:
        for rel in chosen:
            records = db.get_records(rel_ids=[rel])
            sample = records.sample(n=min(per_realisation, len(records)), random_state=seed)
            ids = sample.index.tolist()
            psa = db.get_psa(record_int_ids=ids)
            scalars = db.get_scalars(record_int_ids=ids)
            fas = db.get_fas(record_int_ids=ids) if content.fas else None
            ims = read_ims(manifest[rel].im_path, content, grids)
            row_of = {site: i for i, site in enumerate(ims.stations)}
            for record_int_id, record in sample.iterrows():
                i, component = row_of[record["site_id"]], record["component"]
                where = f"{rel} {record['site_id']} {component}"
                if component in content.psa_components:
                    stored = psa.loc[record_int_id].to_numpy(dtype=np.float32)
                    if not np.array_equal(stored, ims.psa[component][i]):
                        raise AssertionError(f"{where}: pSA differs from the h5")
                    for im in schema.SCALAR_IMS:
                        value = scalars.loc[record_int_id, im]
                        expected = ims.scalars[im].get(component)
                        if expected is None:
                            if not pd.isna(value):
                                raise AssertionError(f"{where}: {im} should be NULL")
                        elif np.float32(value) != expected[i]:
                            raise AssertionError(f"{where}: {im} differs from the h5")
                elif record_int_id in psa.index or record_int_id in scalars.index:
                    raise AssertionError(f"{where}: should have no pSA or scalar IMs")
                if fas is not None:
                    if component in content.fas_components:
                        stored = fas.loc[record_int_id].to_numpy(dtype=np.float32)
                        if not np.array_equal(stored, ims.fas[component][i]):
                            raise AssertionError(f"{where}: FAS differs from the h5")
                    elif record_int_id in fas.index:
                        raise AssertionError(f"{where}: should have no FAS")
                checked += 1
    return checked


def peak_line(row: ManifestRow, peak: dict[str, tuple[float, str]]) -> str:
    """One realisation's largest PGA and PGV (geom), with where they are."""
    (pga, pga_site), (pgv, pgv_site) = peak["PGA"], peak["PGV"]
    return f"{row.label}: PGA {pga:.3f} g at {pga_site}; PGV {pgv:.1f} cm/s at {pgv_site}"


def print_peaks(peaks: Sequence[tuple[ManifestRow, dict[str, tuple[float, str]]]]) -> None:
    """The realisations with the largest PGA, for a human to judge before the database is shared."""
    ranked = sorted(peaks, key=lambda item: item[1]["PGA"][0], reverse=True)
    print(f"top {min(N_TOP_PEAKS, len(ranked))} realisations by max PGA (geom):", flush=True)
    for row, peak in ranked[:N_TOP_PEAKS]:
        print(f"  {peak_line(row, peak)}", flush=True)


def build_database(
    manifest_path: Path,
    stations_input: Path,
    nshm_csv: Path,
    out_path: Path,
    harvested_at: str,
    script_commit: str,
    content: Content = DEFAULT_CONTENT,
    memory_limit: str | None = None,
    expect_flagged: int | None = None,
) -> dict[str, int]:
    """Build the database at out_path and return its table row counts.

    Writes `<out_path>.partial` and renames it only once validation, the row
    counts, the site_event flags and the spot check have all passed. A failure
    leaves the .partial behind for inspection, and the next run refuses to
    start until it is removed.
    """
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    partial = out_path.with_name(out_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"{partial} exists: a previous run died; remove it first")

    rows = read_manifest(manifest_path)
    by_event = group_by_event(rows)
    stations = load_stations_input(stations_input)
    nshm = load_nshm_attributes(nshm_csv)
    missing = sorted(set(by_event) - set(nshm.index), key=int)
    if missing:
        raise ValueError(f"ruptures missing from {nshm_csv}: {missing}")

    grids = read_grids(rows[0].im_path, content)
    db = IMDB.create(
        partial,
        periods=list(grids.periods),
        frequencies=list(grids.frequencies),
        components=content.components,
        db_meta={
            **DB_META,
            **content_meta(content, grids),
            "n_realisations": str(len(rows)),
            "n_pilot_realisations": str(sum(row.pilot for row in rows)),
            "im_calc_harvested_at": harvested_at,
            "build_manifest_md5": md5_of(manifest_path),
            "ingest_script_commit": script_commit,
        },
    )
    counts = dict.fromkeys(
        ("events", "realisations", "sites", "site_event", "records", "psa_ims", "scalars_ims", "fas_ims"), 0
    )
    n_flagged = 0
    seeds_seen: dict[str, str] = {}
    peaks: list[tuple[ManifestRow, dict[str, tuple[float, str]]]] = []
    started = time.monotonic()
    try:
        if memory_limit:
            db.con.raw_sql(f"SET memory_limit = '{memory_limit}'")
        sites = SiteRegistry(stations)
        for index, (event_id, group) in enumerate(by_event.items(), start=1):
            loaded = []
            for row in group:
                realisation, ims = load_realisation(row, content, grids)
                seeds = json.dumps(realisation["seeds"], sort_keys=True)
                if seeds in seeds_seen:
                    raise ValueError(f"{row.label}: the same seeds as {seeds_seen[seeds]}")
                seeds_seen[seeds] = row.rel_id
                check_finite(row.label, ims)
                loaded.append(Loaded(row, realisation, ims))
                peaks.append((row, ims.peaks))
                print(f"peak {peak_line(row, ims.peaks)}", flush=True)

            mains = [x for x in loaded if not x.row.pilot]
            pilots = [x for x in loaded if x.row.pilot]
            defining = mains[0] if mains else pilots[0]
            event = build_event(
                event_id,
                defining.realisation,
                [x.realisation for x in loaded],
                nshm.loc[event_id],
                MAIN_RELEASE if mains else PILOT_RELEASE,
            )
            realisation_rows = []
            for x in loaded:
                realisation_row = build_realisation(x.row, x.realisation)
                check_against_h5(x.row.label, realisation_row, x.ims.attrs)
                realisation_rows.append(realisation_row)
            for x in mains[1:]:
                check_matches_first_realisation(x.row.label, mains[0].row.label, x.ims, mains[0].ims)
            new_sites = pd.concat([sites.new_sites(x.row.label, x.ims) for x in loaded], ignore_index=True)
            site_event, flagged = build_site_event(
                event_id, mains[0].ims if mains else None, pilots[0].ims if pilots else None
            )

            db.add_events(pd.DataFrame([event]))
            db.add_realisations(pd.DataFrame(realisation_rows))
            if len(new_sites):
                db.add_sites(new_sites)
            db.add_site_event(site_event)
            for x in loaded:
                for records, psa, fas in build_records(x.row.rel_id, x.ims, content):
                    db.add_records(records, pSA=psa, FAS=fas)
                n = len(x.ims.stations)
                counts["records"] += n * len(content.components)
                counts["psa_ims"] += n * len(content.psa_components)
                counts["scalars_ims"] += n * len(content.psa_components)
                counts["fas_ims"] += n * len(content.fas_components)

            release_memtables(db)
            counts["events"] += 1
            counts["realisations"] += len(loaded)
            counts["sites"] += len(new_sites)
            counts["site_event"] += len(site_event)
            n_flagged += flagged
            if index % 25 == 0 or index == len(by_event):
                print(
                    f"{index}/{len(by_event)} events, {counts['records']} records, "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )

        problems = db.validate()
        if problems:
            raise RuntimeError(f"validate(): {problems}")
        check_row_counts(db, counts)
        check_site_event_matches_records(db)
        check_flags(db, n_flagged)
        if expect_flagged is not None and n_flagged != expect_flagged:
            raise RuntimeError(f"{n_flagged} site_event rows carry the pilot's distances, expected {expect_flagged}")
    finally:
        db.close()

    checked = spot_check(partial, {row.rel_id: row for row in rows}, content, grids)
    print(
        f"validate() clean; row counts match; {n_flagged} site_event rows carry the "
        f"pilot's distances; spot check: {checked} records identical to their h5",
        flush=True,
    )
    print_peaks(peaks)
    partial.rename(out_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="build manifest CSV from build_imdb_manifest.py")
    parser.add_argument("stations_input", type=Path, help="the campaign's stations_input.ll (lon lat name)")
    parser.add_argument("nshm_csv", type=Path, help="campaign_nshm_ruptures.csv from export_nshm_rupture_attributes.py")
    parser.add_argument("out", type=Path, help="output .duckdb; must not exist")
    parser.add_argument("--harvested-at", required=True, help="when the manifest was built from cylc, UTC ISO 8601")
    parser.add_argument("--script-commit", required=True, help="git commit of this script, for db_meta")
    parser.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help=f"comma-separated, from {','.join(schema.COMPONENTS)}; eas needs --fas (default: %(default)s)",
    )
    parser.add_argument(
        "--psa-periods",
        choices=("default", "all"),
        default="default",
        help="default: the 25 in PSA_PERIODS; all: every one of im-calc's 111",
    )
    parser.add_argument("--fas", action="store_true", help="add FAS on im-calc's full frequency grid")
    parser.add_argument(
        "--expect-flagged",
        type=int,
        help="fail unless exactly this many site_event rows carry the pilot's distances",
    )
    parser.add_argument(
        "--memory-limit",
        help="DuckDB memory_limit, e.g. 12GB. Without it DuckDB sizes itself from "
        "the node's RAM, not the job's memory allocation.",
    )
    args = parser.parse_args()
    content = Content(
        components=tuple(args.components.split(",")),
        all_periods=args.psa_periods == "all",
        fas=args.fas,
    )

    print(f"ingesting {args.manifest} into {args.out}: {content}", flush=True)
    counts = build_database(
        args.manifest,
        args.stations_input,
        args.nshm_csv,
        args.out,
        harvested_at=args.harvested_at,
        script_commit=args.script_commit,
        content=content,
        memory_limit=args.memory_limit,
        expect_flagged=args.expect_flagged,
    )
    print("row counts:", counts)


if __name__ == "__main__":
    main()
