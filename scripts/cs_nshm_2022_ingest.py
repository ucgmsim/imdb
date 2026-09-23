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

cs_nshm_2022 simulated New Zealand NSHM 2022 crustal ruptures with the ucgmsim
workflow on BSC's MareNostrum 5. In the database each event is one rupture,
and each realisation is one simulation of it.

This script is longer than most ingests for two reasons. Two simulation
campaigns become one dataset, and it ran unattended as a Slurm job, so it
checks its own output before publishing it. Sections 2 to 4 are the ingest
itself; most of section 5 is that checking.

Notes
-----
Inputs:

- A build manifest CSV listing every realisation to ingest, with its rel_id,
  pilot flag and file paths. cs_nshm_2022's scripts/build_imdb_manifest.py
  writes it (https://github.com/ucgmsim/cs_nshm_2022).
- For each realisation, the `intensity_measures.h5` written by the workflow's
  im-calc (IMs, distances, vs30 and z1.0/z2.5 at each station: netCDF4, one
  group per IM, one dataset per component) and its `realisation.json`
  (source, rupture propagation and simulation domain).
- The campaign's `stations_input.ll`, for each station's canonical location.
- Each rupture's NSHM magnitude and annual rate, in a CSV exported from
  nshmdb by cs_nshm_2022's scripts/export_nshm_rupture_attributes.py.

What is specific to cs_nshm_2022:

Code that exists only because of the points below is marked `# cs_nshm_2022:`,
so `grep -n "cs_nshm_2022:" cs_nshm_2022_ingest.py` lists what another dataset
can drop.

- Two campaigns. The main campaign simulated up to two realisations per
  rupture, each with its own randomly drawn magnitude, hypocentre, rupture
  propagation and slip. A pilot simulated one realisation for each of 293
  ruptures, at the scaling relation's central magnitude and on fault geometry
  from an earlier release of the NSHM fault database. The manifest numbers each
  event's realisations R1..Rn with no gaps, the pilot's last.
- One geometry per event. An event's geometry, and the distances stored for
  it, come from its first main realisation, or from the pilot where the main
  campaign has none. At a site only the pilot reached, site_event holds the
  pilot's own distances, flagged in its metadata. Each pilot realisation also
  carries its own fault planes.
- Event ids are the rupture's crustal `nshm_id` in nshmdb, NOT its `rupture_id`.
- Stations. Each simulation moved its stations onto its own computational
  grid, so the coordinates stored are the canonical ones from
  stations_input.ll. Real stations have 3-4 character codes; the virtual
  grid's have 7.
- im-calc's pSA grid has 111 periods, with none between 15 s and 20 s.
- nshmdb is not available on BSC, hence the NSHM CSV.

Layout:

1. What goes into the database: the content options.
2. Reading the inputs.
3. Building the rows of each table.
4. The build: each event is read, turned into rows and written, then the
   database is checked and published.
5. Checks.
6. Command line.

By default the database is trimmed for size: only the geom and rotd50
components, pSA on 25 of im-calc's 111 periods, and no FAS. --components,
--psa-periods all and --fas put the rest in. Empirical-GMM records are never
included.

Every magnitude is BoldM (Hanks & Kanamori 1979 eq. 7), the group convention.
Check which convention a script you copy from uses; some convert to Mw.
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

# ---- Constants -----------------------------------------------------------------------

# pSA periods kept by default: 0.1 s steps to 1 s, then 1 s steps.
# cs_nshm_2022: im-calc computed no periods from 16 to 19 s, and 20 s is its last.
PSA_PERIODS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 20.0,
)  # fmt: skip
DEFAULT_COMPONENTS = ("geom", "rotd50")
# Components im-calc writes for pSA, PGA, PGV and PGD, and for FAS.
PSA_COMPONENTS = ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")
FAS_COMPONENTS = ("000", "090", "ver", "geom", "eas")
# cs_nshm_2022: every realisation's im-calc used this pSA grid, so any other
# length means a file from a different configuration.
N_IM_CALC_PERIODS = 111
SITE_FIELDS = ("vs30", "z1pt0", "z2pt5")
DISTANCE_FIELDS = ("rrup", "rjb", "rx", "ry")
TABLES = ("events", "realisations", "sites", "site_event", "records", "psa_ims", "scalars_ims", "fas_ims")
# How closely realisation.json must reproduce the h5's own attributes. Measured
# over all 221 realisations done on 2026-09-21: magnitude identical, hypocentre
# within 3e-14 degrees.
TOLERANCES = {"magnitude": 1e-9, "hypo_lat": 1e-9, "hypo_lon": 1e-9, "hypo_depth": 1e-6}

# cs_nshm_2022: the build manifest's columns.
MANIFEST_FIELDS = (
    "rel_id", "event_id", "realisation", "pilot", "campaign", "original_id", "im_path", "realisation_path",
)  # fmt: skip
# cs_nshm_2022: site_event.metadata on a row whose distances come from the
# pilot's geometry although the event's geometry is the main campaign's.
PILOT_DISTANCES = json.dumps({"fault_geometry": "pilot"})
# cs_nshm_2022: the NSHM fault database releases behind each campaign's geometry.
MAIN_RELEASE = "v2026.08.3"
MAIN_NSHMDB_SHA256 = "3fe692cb9b22c769b6a9baa6a526b4eed989bd61ec054a344f0e74d82855bc4e"
PILOT_RELEASE = "pre-v2026.08"
PILOT_NSHMDB_SHA256 = "00e256480618cd15e11fbf744037d037bf3fc2d523fb977ee30e0b84a640bc57"
# cs_nshm_2022: how many realisations the build log lists by largest PGA.
N_TOP_PEAKS = 20


# ---- 1. What goes into the database --------------------------------------------------
#
# Content is what the command-line options choose; Grids are the pSA periods and
# FAS frequencies the database is created with, which every record then shares.


@dataclass(frozen=True)
class Content:
    """Which components, pSA periods and spectra go into the database.

    Attributes
    ----------
    components : tuple of str
        Which components to include, e.g. `"geom"`, `"rotd50"`, `"eas"`.
    all_periods : bool
        All 111 of im-calc's pSA periods, instead of the 25 in PSA_PERIODS.
    fas : bool
        FAS, on im-calc's full frequency grid, for the components that have it.
    """

    components: tuple[str, ...] = DEFAULT_COMPONENTS
    all_periods: bool = False
    fas: bool = False

    def __post_init__(self) -> None:
        """Raise unless the chosen components are non-empty, unique, known and consistent with fas.

        Raises
        ------
        ValueError
            If `components` is empty, lists a component twice, includes a
            component that is not one of PSA_COMPONENTS or FAS_COMPONENTS,
            includes `eas` without `fas`, or `fas` is set without any
            component that has FAS.
        """
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
        """The chosen components that have pSA and the scalar IMs: all but eas.

        Returns
        -------
        tuple of str
            The chosen components other than `eas`.
        """
        return tuple(c for c in self.components if c in PSA_COMPONENTS)

    @property
    def fas_components(self) -> tuple[str, ...]:
        """The chosen components that get FAS: none without --fas.

        Returns
        -------
        tuple of str
            The chosen components with FAS, empty unless `fas` is set.
        """
        return tuple(c for c in self.components if self.fas and c in FAS_COMPONENTS)


@dataclass(frozen=True)
class Grids:
    """The pSA periods and FAS frequencies the database is created with.

    Attributes
    ----------
    periods : tuple of float
        Response spectral periods, in seconds, that pSA records are indexed by.
    frequencies : tuple of float
        Frequencies, in Hz, that FAS records are indexed by; empty without FAS.
    """

    periods: tuple[float, ...]
    frequencies: tuple[float, ...]


DEFAULT_CONTENT = Content()
DEFAULT_GRIDS = Grids(periods=PSA_PERIODS, frequencies=())


def read_grids(h5_path: Path, content: Content) -> Grids:
    """The database's grids, taken from one realisation's h5 where the content needs its full grid.

    Parameters
    ----------
    h5_path : Path
        Path to one realisation's intensity_measures.h5.
    content : Content
        Which components, periods and spectra to include.

    Returns
    -------
    Grids
        The periods and frequencies the database should be created with.
    """
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
    """db_meta entries describing what the options put in.

    Parameters
    ----------
    content : Content
        Which components, periods and spectra were chosen.
    grids : Grids
        The database's pSA periods and FAS frequencies.

    Returns
    -------
    dict of str to str
        The `psa_period_selection` and `fas` entries for db_meta.
    """
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


# ---- 2. Reading the inputs -----------------------------------------------------------
#
# The manifest says what to ingest. Each realisation's intensity_measures.h5 and
# realisation.json hold its data; the station list and the NSHM CSV are shared
# by the whole campaign. The checks run on what is read are in section 5.


def rel_id(event_id: str, n: int) -> str:
    """The realisation's stable id.

    Parameters
    ----------
    event_id : str
        The rupture's nshm_id.
    n : int
        The realisation number within the event, starting at 1.

    Returns
    -------
    str
        The event id and realisation number joined as `<event_id>_R<n>`.
    """
    return f"{event_id}_R{n}"


# cs_nshm_2022: the two campaigns' results sit in different run directories with
# different layouts, and their realisations are renumbered to be one dataset. A
# manifest names each realisation's files, so this script knows nothing about
# either layout.
@dataclass(frozen=True)
class ManifestRow:
    """One realisation to ingest, as listed in the build manifest.

    Attributes
    ----------
    rel_id : str
        The realisation's stable id, `rel_id(event_id, realisation)`.
    event_id : str
        The rupture's nshm_id, as an unpadded integer string.
    realisation : int
        The realisation number within the event, starting at 1.
    pilot : bool
        Whether this realisation is from the pilot campaign.
    campaign : str
        Which simulation campaign; for checks only, never stored.
    original_id : str
        Where the result came from; for messages only, never stored.
    im_path : Path
        Path to the realisation's intensity_measures.h5.
    realisation_path : Path
        Path to the realisation's realisation.json.
    """

    rel_id: str
    event_id: str
    realisation: int
    pilot: bool
    campaign: str
    original_id: str
    im_path: Path
    realisation_path: Path

    @property
    def label(self) -> str:
        """The realisation's id and original id, for messages.

        Returns
        -------
        str
            The realisation formatted as `<rel_id> (<original_id>)`.
        """
        return f"{self.rel_id} ({self.original_id})"


def read_manifest(path: Path) -> list[ManifestRow]:
    """The manifest's rows in ingest order: events by integer id, then realisations.

    Parameters
    ----------
    path : Path
        Path to the build manifest CSV.

    Returns
    -------
    list of ManifestRow
        The manifest's rows, sorted by event id and realisation number.

    Raises
    ------
    ValueError
        If the CSV is missing a required column, or a row's `pilot` field is
        not "true" or "false".
    FileNotFoundError
        If any listed `im_path` or `realisation_path` does not exist.
    """
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


def group_by_event(rows: Sequence[ManifestRow]) -> dict[str, list[ManifestRow]]:
    """Rows per event, keeping the input order.

    Parameters
    ----------
    rows : Sequence of ManifestRow
        Manifest rows to group.

    Returns
    -------
    dict of str to list of ManifestRow
        Each event id mapped to its rows, in the order they appear in `rows`.
    """
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_id, []).append(row)
    return grouped


# cs_nshm_2022: each simulation moved its stations onto its own computational
# grid (about 0.1 km) and wrote those coordinates into its h5, so they differ
# between simulations. sites stores each station's canonical location instead.
def load_stations_input(path: Path) -> pd.DataFrame:
    """Canonical station coordinates (`lon lat name` per line), indexed by name.

    Parameters
    ----------
    path : Path
        Path to the campaign's stations_input.ll.

    Returns
    -------
    pd.DataFrame
        Its `lon` and `lat` columns, indexed by `site_id`.

    Raises
    ------
    ValueError
        If the file lists the same station name twice.
    """
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=["lon", "lat", "site_id"], dtype={"site_id": str}
    )
    duplicated = df["site_id"][df["site_id"].duplicated()]
    if len(duplicated):
        raise ValueError(f"{path}: duplicate station names, e.g. {list(duplicated[:5])}")
    return df.set_index("site_id")


def is_real_station(site_id: str) -> bool:
    """GeoNet station codes are 3-4 characters; the virtual grid's codes are 7.

    Parameters
    ----------
    site_id : str
        A station's canonical name.

    Returns
    -------
    bool
        Whether the name's length marks it as a real, not virtual-grid, station.
    """
    # cs_nshm_2022: nothing in the inputs marks a station as real, so the name
    # length decides. Check how your own station list names its stations.
    return len(site_id) != 7


# cs_nshm_2022: nshmdb is not available on BSC, where this ran, so each
# rupture's NSHM attributes come from a CSV exported beforehand. The CSV is
# indexed by the crustal nshm_id, which is this dataset's event_id; nshmdb's own
# rupture_id is a different number and goes into events.metadata.
def load_nshm_attributes(path: Path) -> pd.DataFrame:
    """Rows of export_nshm_rupture_attributes.py's CSV, indexed by rupture id.

    Parameters
    ----------
    path : Path
        Path to the exported NSHM attributes CSV.

    Returns
    -------
    pd.DataFrame
        One row per rupture, indexed by `rupture_id`.

    Raises
    ------
    ValueError
        If the CSV's `nshmdb_sha256` values are not all MAIN_NSHMDB_SHA256,
        i.e. they do not all come from the release DB_META documents.
    """
    nshm = pd.read_csv(path, dtype={"rupture_id": str}).set_index("rupture_id")
    if set(nshm["nshmdb_sha256"]) != {MAIN_NSHMDB_SHA256}:
        raise ValueError(
            f"{path}: NSHM attributes must all come from {MAIN_RELEASE} "
            f"(sha256 {MAIN_NSHMDB_SHA256}), found {sorted(set(nshm['nshmdb_sha256']))}"
        )
    return nshm


@dataclass
class RealisationIMs:
    """One realisation's intensity_measures.h5, reduced to what this database keeps.

    Attributes
    ----------
    stations : np.ndarray
        Station names, str.
    site : dict of str to np.ndarray
        SITE_FIELDS per station.
    distances : dict of str to np.ndarray
        DISTANCE_FIELDS per station, km.
    psa : dict of str to np.ndarray
        Component -> (n_stations, n_periods) float32, for Content.psa_components.
    fas : dict of str to np.ndarray
        Component -> (n_stations, n_frequencies) float32, for Content.fas_components.
    n_frequencies : int
        Length of the database's FAS grid; 0 without FAS.
    scalars : dict of str to dict of str to np.ndarray
        Scalar IM -> component -> (n_stations,) float32; rotd-undefined IMs omitted.
    attrs : dict of str to float
        The file's magnitude and hypocentre attributes (TOLERANCES keys).
    peaks : dict of str to tuple of (float, str)
        PGA and PGV (geom) -> (largest value, its station), whatever the database keeps.
    """

    stations: np.ndarray
    site: dict[str, np.ndarray]
    distances: dict[str, np.ndarray]
    psa: dict[str, np.ndarray]
    fas: dict[str, np.ndarray]
    n_frequencies: int
    scalars: dict[str, dict[str, np.ndarray]]
    attrs: dict[str, float]
    peaks: dict[str, tuple[float, str]]


def _decode(values: np.ndarray) -> np.ndarray:
    """Decode any bytes elements of an array to str, leaving others unchanged.

    Parameters
    ----------
    values : np.ndarray
        Array whose elements may be bytes, as h5py returns for HDF5 strings.

    Returns
    -------
    np.ndarray
        Object array with every bytes element decoded to str.
    """
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def _peak(values: np.ndarray, stations: np.ndarray) -> tuple[float, str]:
    """The largest value in `values` and the station it belongs to, ignoring NaN.

    Parameters
    ----------
    values : np.ndarray
        One value per station.
    stations : np.ndarray
        Station names in the same order as `values`, str or bytes.

    Returns
    -------
    value : float
        The largest value.
    station : str
        The station it belongs to, decoded to str.
    """
    i = int(np.nanargmax(values))
    station = stations[i]
    return float(values[i]), station.decode() if isinstance(station, bytes) else str(station)


def select_period_indices(
    available: np.ndarray, wanted: Sequence[float] = PSA_PERIODS
) -> np.ndarray:
    """Column of each wanted period in im-calc's period grid.

    Parameters
    ----------
    available : np.ndarray
        The full period grid read from one realisation's h5.
    wanted : Sequence of float, optional
        Periods to find columns for (default PSA_PERIODS).

    Returns
    -------
    np.ndarray
        Column index in `available` of each period in `wanted`, in order.

    Raises
    ------
    ValueError
        If any wanted period does not match exactly one available one.
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


def read_ims(
    h5_path: Path, content: Content = DEFAULT_CONTENT, grids: Grids = DEFAULT_GRIDS
) -> RealisationIMs:
    """Read the parts of one intensity_measures.h5 this database keeps.

    Parameters
    ----------
    h5_path : Path
        Path to the realisation's intensity_measures.h5.
    content : Content, optional
        Which components, periods and spectra to keep (default DEFAULT_CONTENT).
    grids : Grids, optional
        The database's pSA periods and FAS frequencies (default DEFAULT_GRIDS).

    Returns
    -------
    RealisationIMs
        The stations, site fields, distances, IMs and peaks this database keeps.

    Raises
    ------
    ValueError
        If the file's pSA period count is not `N_IM_CALC_PERIODS`, an IM or
        FAS group lists stations in a different order from pSA, a wanted
        component is missing where it is not rotd-undefined, or the file's
        FAS frequency grid differs from the database's.
    """
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


@dataclass
class Loaded:
    """One realisation read from disk.

    Attributes
    ----------
    row : ManifestRow
        The manifest row this realisation was loaded from.
    realisation : dict
        The parsed realisation.json.
    ims : RealisationIMs
        The parts of intensity_measures.h5 this database keeps.
    """

    row: ManifestRow
    realisation: dict
    ims: RealisationIMs


def load_realisation(row: ManifestRow, content: Content, grids: Grids) -> tuple[dict, RealisationIMs]:
    """Read one realisation's realisation.json and h5, naming the realisation in any error.

    Parameters
    ----------
    row : ManifestRow
        The manifest row to load.
    content : Content
        Which components, periods and spectra to keep.
    grids : Grids
        The database's pSA periods and FAS frequencies.

    Returns
    -------
    realisation : dict
        The parsed realisation.json.
    ims : RealisationIMs
        The parts of intensity_measures.h5 this database keeps.

    Raises
    ------
    KeyError, OSError, ValueError
        Any of these from reading either file, re-raised with the
        realisation's label at the front of the message; ValueError also if
        realisation.json's `metadata.name` is not `"Rupture <event_id>"`.
    """
    try:
        realisation = json.loads(row.realisation_path.read_text())
        # cs_nshm_2022: the workflow names each realisation "Rupture <nshm_id>",
        # which catches a manifest row that points at another event's files.
        name = realisation["metadata"]["name"]
        if name != f"Rupture {row.event_id}":
            raise ValueError(f"realisation.json describes {name!r}, not event {row.event_id}")
        return realisation, read_ims(row.im_path, content, grids)
    except (KeyError, OSError, ValueError) as error:
        raise type(error)(f"{row.label}: {error}") from error


# ---- 3. Building the rows ------------------------------------------------------------
#
# One builder per table. The geometry helpers are the same as in the other
# ingest scripts that read realisation.json.


def fault_geometries(realisation: dict) -> dict[str, Fault]:
    """Each named source geometry in realisation.json, as a Fault.

    Parameters
    ----------
    realisation : dict
        The parsed realisation.json.

    Returns
    -------
    dict of str to Fault
        Each `sources.source_geometries` entry's name mapped to its Fault,
        built from its corners' latitude, longitude and depth.
    """
    return {
        name: Fault.from_corners(
            np.array([[c["latitude"], c["longitude"], c["depth"]] for c in entry["corners"]]).reshape(-1, 4, 3)
        )
        for name, entry in realisation["sources"]["source_geometries"].items()
    }


def initial_fault_name(causality_tree: dict[str, str | None]) -> str:
    """The name of the fault the rupture starts on: the one entry with no parent.

    Parameters
    ----------
    causality_tree : dict of str to str or None
        Each fault name mapped to its parent's name, or None for the fault
        the rupture starts on.

    Returns
    -------
    str
        The initiating fault's name.
    """
    return next(name for name, parent in causality_tree.items() if parent is None)


def source_wkt(geometries: dict[str, Fault]) -> str:
    """The geometries' fault planes as a 2D MULTIPOLYGON WKT, for events.source_wkt.

    Parameters
    ----------
    geometries : dict of str to Fault
        Each source's fault geometry, as from `fault_geometries`.

    Returns
    -------
    str
        A MULTIPOLYGON WKT with one polygon per plane, corners as (lon, lat).
    """
    planes = [plane for f in geometries.values() for plane in f.planes]
    return MultiPolygon([Polygon(p.corners[:, [1, 0]]) for p in planes]).wkt


def trace_wkt(geometries: dict[str, Fault]) -> str:
    """The geometries' fault plane top edges as a MULTILINESTRING WKT, for events.trace_wkt.

    Parameters
    ----------
    geometries : dict of str to Fault
        Each source's fault geometry, as from `fault_geometries`.

    Returns
    -------
    str
        A MULTILINESTRING WKT with one line per plane's top edge, as (lon, lat).
    """
    planes = [plane for f in geometries.values() for plane in f.planes]
    return MultiLineString([LineString(p.corners[:2, [1, 0]]) for p in planes]).wkt


def fault_geometry_wkt(realisation: dict) -> str:
    """The realisation's own fault planes, straight from realisation.json.

    A MULTIPOLYGON Z with one polygon per plane, corners as (lon, lat, depth
    in km). Unlike events.source_wkt it carries depth, so distances can be
    recomputed from it alone.

    Parameters
    ----------
    realisation : dict
        The parsed realisation.json.

    Returns
    -------
    str
        A MULTIPOLYGON Z WKT built directly from `sources.source_geometries`.
    """
    planes = []
    for entry in realisation["sources"]["source_geometries"].values():
        corners = [[c["longitude"], c["latitude"], c["depth"] / 1000] for c in entry["corners"]]
        planes.extend(Polygon(plane) for plane in np.array(corners).reshape(-1, 4, 3))
    return MultiPolygon(planes).wkt


def domain_polygon(realisation: dict) -> Polygon:
    """The realisation's simulation domain, from realisation.json.

    Parameters
    ----------
    realisation : dict
        The parsed realisation.json.

    Returns
    -------
    Polygon
        The domain, with its corners as (lon, lat).
    """
    corners = realisation["domain"]["domain"]
    return Polygon([(c["longitude"], c["latitude"]) for c in corners])


def domain_wkt(realisations: Sequence[dict]) -> str:
    """The union of the realisations' simulation domains, so it contains every site with data.

    Parameters
    ----------
    realisations : Sequence of dict
        Parsed realisation.json documents to union.

    Returns
    -------
    str
        The union of `domain_polygon` over `realisations`, as WKT.
    """
    return unary_union([domain_polygon(r) for r in realisations]).wkt


def total_magnitude(realisation: dict) -> float:
    """The realisation's total magnitude, BoldM: segment moments summed, converted back.

    Parameters
    ----------
    realisation : dict
        The parsed realisation.json.

    Returns
    -------
    float
        The BoldM magnitude equivalent to the sum of the segments' moments.
    """
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

    cs_nshm_2022: with one campaign, `defining` would be any realisation and
    the domain its own. Here the pilot's geometry and domain differ from the
    main campaign's, so the caller picks `defining`, and the union keeps every
    site with data inside the event's domain.

    Parameters
    ----------
    event_id : str
        The rupture's nshm_id.
    defining : dict
        The parsed realisation.json whose geometry and tect_type the event uses.
    realisations : Sequence of dict
        Every realisation of this event, parsed, for the domain union.
    nshm : pd.Series
        The rupture's row from `load_nshm_attributes`.
    release : str
        The NSHM fault database release `defining`'s geometry came from.

    Returns
    -------
    dict
        One row for the `events` table.

    Raises
    ------
    ValueError
        If `defining`'s `empirical.tect_type` is not one of `schema.TECT_TYPES`.
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
    """The realisations row, from the realisation's own realisation.json and geometry.

    Parameters
    ----------
    row : ManifestRow
        The manifest row this realisation was loaded from.
    realisation : dict
        The parsed realisation.json.

    Returns
    -------
    dict
        One row for the `realisations` table.
    """
    geometries = fault_geometries(realisation)
    causality_tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    initial_fault = initial_fault_name(causality_tree)
    hypocentre_sd = realisation["rupture_propagation"]["hypocentre"]
    hypo_lat, hypo_lon, hypo_depth_m = geometries[initial_fault].fault_coordinates_to_wgs_depth_coordinates(
        np.array([hypocentre_sd["s"], hypocentre_sd["d"]])
    )
    metadata = {
        "pilot": row.pilot,
        "seeds": realisation["seeds"],
        "segment_magnitudes_boldm": realisation["magnitudes"]["magnitudes"],
        "rupture_causality_tree": causality_tree,
        "duration_s": realisation["domain"]["duration"],
    }
    if row.pilot:
        # cs_nshm_2022: the pilot's geometry comes from an earlier NSHM database
        # release, so it is not the event's wherever the main campaign also
        # simulated the rupture.
        metadata["fault_geometry_wkt"] = fault_geometry_wkt(realisation)
    return {
        "rel_id": row.rel_id,
        "event_id": row.event_id,
        "magnitude": total_magnitude(realisation),
        "rake": realisation["rakes"]["rakes"][initial_fault],
        "hypo_lat": float(hypo_lat),
        "hypo_lon": float(hypo_lon),
        "hypo_depth": float(hypo_depth_m) / 1000,
        "metadata": json.dumps(metadata),
    }


class SiteRegistry:
    """The sites added so far, with the vs30/z1pt0/z2pt5 first seen for each.

    Parameters
    ----------
    stations_input : pd.DataFrame
        Canonical station coordinates, as from `load_stations_input`.
    """

    def __init__(self, stations_input: pd.DataFrame) -> None:
        """Start with no sites seen yet.

        Parameters
        ----------
        stations_input : pd.DataFrame
            Canonical station coordinates, as from `load_stations_input`.
        """
        self.stations_input = stations_input
        self.seen: pd.DataFrame | None = None

    def new_sites(self, label: str, ims: RealisationIMs) -> pd.DataFrame:
        """Rows for stations not added yet; raise if a known one disagrees.

        Parameters
        ----------
        label : str
            The realisation's label, for the error message.
        ims : RealisationIMs
            The realisation's IMs, whose stations and site fields are checked
            against and added to the registry.

        Returns
        -------
        pd.DataFrame
            One `sites` row per station in `ims` not already registered.

        Raises
        ------
        ValueError
            If a station already seen has different site fields this time, or
            a new station is missing from `stations_input`.
        """
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


def _site_event_rows(
    event_id: str, stations: np.ndarray, distances: dict[str, np.ndarray], metadata: str | None
) -> pd.DataFrame:
    """Assemble site_event rows from parallel station and distance arrays.

    Parameters
    ----------
    event_id : str
        The event these rows belong to.
    stations : np.ndarray
        Station names, one per row.
    distances : dict of str to np.ndarray
        DISTANCE_FIELDS arrays, one value per row, in the same order as `stations`.
    metadata : str or None
        `site_event.metadata` value for every row.

    Returns
    -------
    pd.DataFrame
        One `site_event` row per station.
    """
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

    cs_nshm_2022: with one campaign, this would be one realisation's distances.

    Parameters
    ----------
    event_id : str
        The event these rows belong to.
    main : RealisationIMs or None
        The event's first main-campaign realisation's IMs, or None if it has none.
    pilot : RealisationIMs or None
        The event's pilot realisation's IMs, or None if it has none.

    Returns
    -------
    rows : pd.DataFrame
        One `site_event` row per site the event reaches.
    n_flagged : int
        How many rows carry the pilot's distances.

    Raises
    ------
    ValueError
        If both `main` and `pilot` are None.
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


Batch = tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]


def build_records(rel: str, ims: RealisationIMs, content: Content = DEFAULT_CONTENT) -> list[Batch]:
    """IMDB records (one per station and component), as add_records batches of (records, pSA, FAS).

    `eas` has only FAS, so its records form a batch of their own with no pSA
    and no scalar columns, and add_records writes no psa_ims or scalars_ims
    rows for them. A rotd record's FAS row is all NaN, so it gets no fas_ims row.

    Parameters
    ----------
    rel : str
        The realisation's rel_id.
    ims : RealisationIMs
        The realisation's IMs to turn into records.
    content : Content, optional
        Which components, periods and spectra to include (default DEFAULT_CONTENT).

    Returns
    -------
    list of Batch
        One (records, pSA, FAS) batch per `IMDB.add_records` call needed: the
        pSA components' batch, if any, then eas's, if included.
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


# ---- 4. The build --------------------------------------------------------------------
#
# One event at a time: read its realisations, build its rows, write them. The
# database is written as <out>.partial and renamed only after section 5's
# checks pass, so a failed or killed job never leaves a finished-looking file.

# What the database says about itself. Every dataset needs its own; most of this
# one explains cs_nshm_2022's two campaigns to the people who will query it.
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
        "a median of about 210 m, at most 2.6 km). Each pilot realisation carries its own "
        "fault planes in realisations.metadata.fault_geometry_wkt: a MULTIPOLYGON Z with "
        "one polygon per plane, corners as (longitude, latitude, depth in km)."
    ),
    "distances": (
        "site_event rrup, rjb, rx and ry are computed from the event's own fault geometry, "
        "events.source_wkt (the main campaign's where the event has a main-campaign "
        "realisation, else the pilot's), and have NULL metadata. One exception: in an "
        "event that also has main-campaign realisations, at a site only its pilot "
        "realisation covers, the distances come from the pilot's own geometry "
        "(realisations.metadata.fault_geometry_wkt), which is not events.source_wkt, and "
        f"site_event.metadata is {PILOT_DISTANCES}. Epicentral "
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


def md5_of(path: Path) -> str:
    """Hex md5 of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        The file's contents' hex-encoded md5 digest.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def release_memtables(db: IMDB) -> None:
    """Drop the in-memory tables ibis registers for every insert.

    ibis (12.0) registers each inserted DataFrame with DuckDB as an Arrow table
    and releases them only at interpreter exit, so without this every row
    written stays in memory, outside DuckDB's memory_limit, for the whole build.
    Any large ingest needs it until imdb releases them itself.

    Parameters
    ----------
    db : IMDB
        The database to release memtables from.
    """
    raw = db.con.con
    for (name,) in raw.execute(
        "SELECT view_name FROM duckdb_views() WHERE temporary AND view_name LIKE 'ibis_pandas_memtable_%'"
    ).fetchall():
        raw.unregister(name)


def read_event(
    group: Sequence[ManifestRow], content: Content, grids: Grids, seeds_seen: dict[str, str]
) -> list[Loaded]:
    """Read one event's realisations, checking each on its own.

    Parameters
    ----------
    group : Sequence of ManifestRow
        The event's manifest rows.
    content : Content
        Which components, periods and spectra to keep.
    grids : Grids
        The database's pSA periods and FAS frequencies.
    seeds_seen : dict of str to str
        Every realisation's seeds read so far, mapped to its rel_id, which
        catches the same simulation listed twice under different ids.
        Updated in place with this event's realisations.

    Returns
    -------
    list of Loaded
        The event's realisations, read and checked.

    Raises
    ------
    ValueError
        If a realisation's seeds match one already in `seeds_seen`.
    """
    loaded = []
    for row in group:
        realisation, ims = load_realisation(row, content, grids)
        seeds = json.dumps(realisation["seeds"], sort_keys=True)
        if seeds in seeds_seen:
            raise ValueError(f"{row.label}: the same seeds as {seeds_seen[seeds]}")
        seeds_seen[seeds] = row.rel_id
        check_finite(row.label, ims)
        loaded.append(Loaded(row, realisation, ims))
        # cs_nshm_2022: logged so extreme near-fault values could be judged
        # before the database was shared.
        print(f"peak {peak_line(row, ims.peaks)}", flush=True)
    return loaded


@dataclass
class EventRows:
    """Everything one event adds to the database apart from its records.

    Attributes
    ----------
    event : dict
        The event's `events` row.
    realisations : list of dict
        The event's `realisations` rows.
    sites : pd.DataFrame
        Sites no earlier event reached.
    site_event : pd.DataFrame
        The event's `site_event` rows.
    n_flagged : int
        How many site_event rows carry the pilot's distances.
    """

    event: dict
    realisations: list[dict]
    sites: pd.DataFrame
    site_event: pd.DataFrame
    n_flagged: int


def build_event_rows(event_id: str, loaded: Sequence[Loaded], nshm: pd.Series, sites: SiteRegistry) -> EventRows:
    """The events, realisations, sites and site_event rows for one event.

    Parameters
    ----------
    event_id : str
        The rupture's nshm_id.
    loaded : Sequence of Loaded
        The event's realisations, read and checked.
    nshm : pd.Series
        The rupture's row from `load_nshm_attributes`.
    sites : SiteRegistry
        The sites added by earlier events; updated in place with this event's
        new ones.

    Returns
    -------
    EventRows
        The event's rows for every table but records.
    """
    # cs_nshm_2022: the event's geometry and unflagged distances come from its
    # first main realisation, or from the pilot where the main campaign never
    # simulated the rupture.
    mains = [sim for sim in loaded if not sim.row.pilot]
    pilots = [sim for sim in loaded if sim.row.pilot]
    defining = mains[0] if mains else pilots[0]
    event = build_event(
        event_id,
        defining.realisation,
        [sim.realisation for sim in loaded],
        nshm,
        MAIN_RELEASE if mains else PILOT_RELEASE,
    )
    realisation_rows = []
    for sim in loaded:
        realisation_row = build_realisation(sim.row, sim.realisation)
        check_against_h5(sim.row.label, realisation_row, sim.ims.attrs)
        realisation_rows.append(realisation_row)
    for sim in mains[1:]:
        check_matches_first_realisation(sim.row.label, mains[0].row.label, sim.ims, mains[0].ims)
    new_sites = pd.concat([sites.new_sites(sim.row.label, sim.ims) for sim in loaded], ignore_index=True)
    site_event, n_flagged = build_site_event(
        event_id, mains[0].ims if mains else None, pilots[0].ims if pilots else None
    )
    return EventRows(event, realisation_rows, new_sites, site_event, n_flagged)


def write_event(db: IMDB, rows: EventRows, loaded: Sequence[Loaded], content: Content) -> dict[str, int]:
    """Write one event, and return how many rows each table should have gained.

    Parameters
    ----------
    db : IMDB
        The database to write to, open for writing.
    rows : EventRows
        The event's rows for every table but records.
    loaded : Sequence of Loaded
        The event's realisations, for building and writing their records.
    content : Content
        Which components, periods and spectra to write.

    Returns
    -------
    dict of str to int
        How many rows each table should have gained.
    """
    db.add_events(pd.DataFrame([rows.event]))
    db.add_realisations(pd.DataFrame(rows.realisations))
    if len(rows.sites):
        db.add_sites(rows.sites)
    db.add_site_event(rows.site_event)
    for sim in loaded:
        for records, psa, fas in build_records(sim.row.rel_id, sim.ims, content):
            db.add_records(records, pSA=psa, FAS=fas)
    release_memtables(db)

    n_stations = sum(len(sim.ims.stations) for sim in loaded)
    return {
        "events": 1,
        "realisations": len(rows.realisations),
        "sites": len(rows.sites),
        "site_event": len(rows.site_event),
        "records": n_stations * len(content.components),
        "psa_ims": n_stations * len(content.psa_components),
        "scalars_ims": n_stations * len(content.psa_components),
        "fas_ims": n_stations * len(content.fas_components),
    }


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

    Parameters
    ----------
    manifest_path : Path
        Path to the build manifest CSV.
    stations_input : Path
        Path to the campaign's stations_input.ll.
    nshm_csv : Path
        Path to export_nshm_rupture_attributes.py's CSV.
    out_path : Path
        Path to write the finished database to; must not exist.
    harvested_at : str
        When the manifest was built from cylc, UTC ISO 8601, stored in db_meta.
    script_commit : str
        This script's git commit, stored in db_meta.
    content : Content, optional
        Which components, periods and spectra to include (default DEFAULT_CONTENT).
    memory_limit : str, optional
        DuckDB `memory_limit`, e.g. `"12GB"`. Without it DuckDB sizes itself
        from the node's RAM, not the job's memory allocation.
    expect_flagged : int, optional
        If given, fail unless exactly this many site_event rows carry the
        pilot's distances.

    Returns
    -------
    dict of str to int
        Each table's row count in the finished database.

    Raises
    ------
    FileExistsError
        If `out_path` already exists, or a `.partial` file from a previous,
        unfinished run is still there.
    ValueError
        If the manifest lists a rupture not found in `nshm_csv`.
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
            # cs_nshm_2022: provenance the Slurm job supplies: when the manifest
            # was taken from the workflow, and this script's git commit.
            "im_calc_harvested_at": harvested_at,
            "build_manifest_md5": md5_of(manifest_path),
            "ingest_script_commit": script_commit,
        },
    )
    counts = dict.fromkeys(TABLES, 0)
    n_flagged = 0
    seeds_seen: dict[str, str] = {}
    peaks: list[tuple[ManifestRow, dict[str, tuple[float, str]]]] = []
    started = time.monotonic()
    try:
        if memory_limit:
            db.con.raw_sql(f"SET memory_limit = '{memory_limit}'")
        sites = SiteRegistry(stations)
        for index, (event_id, group) in enumerate(by_event.items(), start=1):
            loaded = read_event(group, content, grids, seeds_seen)
            event_rows = build_event_rows(event_id, loaded, nshm.loc[event_id], sites)
            for table, n in write_event(db, event_rows, loaded, content).items():
                counts[table] += n
            n_flagged += event_rows.n_flagged
            peaks.extend((sim.row, sim.ims.peaks) for sim in loaded)
            if index % 25 == 0 or index == len(by_event):
                print(
                    f"{index}/{len(by_event)} events, {counts['records']} records, "
                    f"{time.monotonic() - started:.0f} s",
                    flush=True,
                )
        verify_database(db, counts, n_flagged, expect_flagged)
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


# ---- 5. Checks -----------------------------------------------------------------------
#
# The ingest ran unattended, so it checks its inputs as it reads them and its
# output before publishing it. Most of these are worth keeping in any ingest;
# those marked cs_nshm_2022 concern the manifest and the pilot's distances.


def check_manifest(rows: Sequence[ManifestRow]) -> None:
    """Raise unless ids are consistent and unique, and every event is R1..Rn with at most one pilot, last.

    The pilot flag must also follow the campaign: no campaign may supply both
    pilot and non-pilot rows.

    Parameters
    ----------
    rows : Sequence of ManifestRow
        The manifest rows to check.

    Raises
    ------
    ValueError
        If `rows` is empty, a campaign supplies both pilot and non-pilot
        rows, an `event_id` is not an unpadded integer, a `rel_id` does not
        match `rel_id(event_id, realisation)`, a `rel_id` is listed twice, an
        event's realisation numbers are not `1..n` with no gaps, an event has
        more than one pilot realisation, or an event's pilot realisation is
        not last.
    """
    if not rows:
        raise ValueError("the manifest lists no realisations")
    # cs_nshm_2022: the pilot flag is decided per campaign, never per realisation.
    pilot_campaigns = {row.campaign for row in rows if row.pilot}
    main_campaigns = {row.campaign for row in rows if not row.pilot}
    if pilot_campaigns & main_campaigns:
        raise ValueError(
            f"campaign {sorted(pilot_campaigns & main_campaigns)} has both pilot and non-pilot rows"
        )
    seen: set[str] = set()
    for row in rows:
        if not row.event_id.isdigit() or str(int(row.event_id)) != row.event_id:
            raise ValueError(f"{row.rel_id}: event_id {row.event_id!r} is not an unpadded integer")
        if row.rel_id != rel_id(row.event_id, row.realisation):
            raise ValueError(
                f"{row.rel_id}: expected {rel_id(row.event_id, row.realisation)} "
                "from event_id and realisation"
            )
        if row.rel_id in seen:
            raise ValueError(f"{row.rel_id}: listed twice")
        seen.add(row.rel_id)
    # cs_nshm_2022: renumbering the two campaigns leaves every event R1..Rn,
    # with the pilot's realisation last.
    for event_id, group in group_by_event(rows).items():
        numbers = sorted(row.realisation for row in group)
        if numbers != list(range(1, len(group) + 1)):
            raise ValueError(f"event {event_id}: realisations {numbers}, expected 1..{len(group)}")
        pilots = [row for row in group if row.pilot]
        if len(pilots) > 1:
            raise ValueError(f"event {event_id}: {len(pilots)} pilot realisations, expected at most 1")
        if pilots and pilots[0].realisation != len(group):
            raise ValueError(
                f"event {event_id}: the pilot is R{pilots[0].realisation}, "
                f"but it must be the last, R{len(group)}"
            )


def check_finite(rel: str, ims: RealisationIMs) -> None:
    """Raise if any kept IM, site or distance value is NaN or infinite.

    Parameters
    ----------
    rel : str
        The realisation's label, for the error message.
    ims : RealisationIMs
        The realisation's IMs to check.

    Raises
    ------
    ValueError
        If any pSA, FAS, scalar IM, site or distance array has a non-finite value.
    """
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


def check_against_h5(rel: str, realisation_row: dict, attrs: dict[str, float]) -> None:
    """Raise unless realisation.json reproduces the h5's magnitude and hypocentre.

    Parameters
    ----------
    rel : str
        The realisation's label, for the error message.
    realisation_row : dict
        The realisation's `realisations` row, built from realisation.json.
    attrs : dict of str to float
        The h5's own magnitude and hypocentre attributes (TOLERANCES keys).

    Raises
    ------
    ValueError
        If any TOLERANCES field differs between `realisation_row` and `attrs`
        by more than its tolerance.
    """
    for key, tolerance in TOLERANCES.items():
        ours, theirs = realisation_row[key], attrs[key]
        if not abs(ours - theirs) <= tolerance:
            raise ValueError(
                f"{rel}: {key} from realisation.json is {ours!r}, but "
                f"intensity_measures.h5 says {theirs!r}"
            )


def check_matches_first_realisation(
    rel: str, first_rel: str, ims: RealisationIMs, first: RealisationIMs
) -> None:
    """Raise unless stations, site fields and distances equal the event's first main realisation's.

    site_event keeps one set of distances per event, so an event's realisations
    must agree on them. Nothing in the workflow enforces that; they agree only
    because they share one fault database and one code version.

    Parameters
    ----------
    rel : str
        This realisation's label, for the error message.
    first_rel : str
        The event's first main realisation's label, for the error message.
    ims : RealisationIMs
        This realisation's IMs.
    first : RealisationIMs
        The event's first main realisation's IMs, to compare against.

    Raises
    ------
    ValueError
        If the station list differs from `first`'s, or any site or distance
        field differs from `first`'s.
    """
    if not np.array_equal(ims.stations, first.stations):
        raise ValueError(f"{rel}: station list differs from {first_rel}")
    for mine, theirs in ((ims.site, first.site), (ims.distances, first.distances)):
        for key, values in mine.items():
            if not np.array_equal(values, theirs[key]):
                raise ValueError(
                    f"{rel}: {key} differs from {first_rel}; site and distance fields "
                    "must be identical across an event's main realisations"
                )


def check_row_counts(db: IMDB, expected: dict[str, int]) -> None:
    """Raise unless every table has exactly the expected number of rows.

    Parameters
    ----------
    db : IMDB
        The database to check, open.
    expected : dict of str to int
        Each table's expected row count.

    Raises
    ------
    RuntimeError
        If a table's row count differs from `expected`.
    """
    for table, n in expected.items():
        found = int(db.con.table(table).count().to_pandas())
        if found != n:
            raise RuntimeError(f"{table}: {found} rows, expected {n}")


# site_event must hold exactly the (site, event) pairs that have records: the
# pairs in site_event but not in records, and those in records but not in site_event.
SITE_EVENT_QUERY = """
WITH pairs AS (SELECT DISTINCT site_int_id, event_int_id FROM records)
SELECT
    (SELECT count(*) FROM site_event ANTI JOIN pairs USING (site_int_id, event_int_id)),
    (SELECT count(*) FROM pairs ANTI JOIN site_event USING (site_int_id, event_int_id))
"""


def check_site_event_matches_records(db: IMDB) -> None:
    """Raise unless site_event has a row for exactly the (site, event) pairs that have records.

    Parameters
    ----------
    db : IMDB
        The database to check, open.

    Raises
    ------
    RuntimeError
        If any site_event row has no matching records, or any (site, event)
        pair with records has no site_event row.
    """
    extra, missing = db.con.raw_sql(SITE_EVENT_QUERY).fetchone()
    if extra or missing:
        raise RuntimeError(f"site_event: {extra} rows with no records, {missing} record pairs with no row")


# cs_nshm_2022: counts the flagged site_event rows three ways, from the tables
# alone: the rows flagged; the (site, event) pairs where only a pilot realisation
# has records in an event that also has main realisations; and the overlap of the two.
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


def check_flags(db: IMDB, n_flagged: int) -> None:
    """Raise unless the flagged site_event rows are exactly the pilot-only sites of shared events.

    Parameters
    ----------
    db : IMDB
        The database to check, open.
    n_flagged : int
        How many site_event rows the build wrote with the pilot's distances.

    Raises
    ------
    RuntimeError
        If the flagged row count, the pilot-only-site count and `n_flagged`
        are not all equal.
    """
    flagged, pilot_only, both = db.con.raw_sql(FLAG_QUERY).fetchone()
    if not flagged == pilot_only == both == n_flagged:
        raise RuntimeError(
            f"site_event flags: {flagged} flagged rows, {pilot_only} pilot-only sites in "
            f"shared events, {both} in both, {n_flagged} written"
        )


def verify_database(db: IMDB, counts: dict[str, int], n_flagged: int, expect_flagged: int | None) -> None:
    """Raise unless the finished database passes every check that needs only its tables.

    Parameters
    ----------
    db : IMDB
        The database to check, open.
    counts : dict of str to int
        Each table's expected row count.
    n_flagged : int
        How many site_event rows the build wrote with the pilot's distances.
    expect_flagged : int or None
        If given, the exact `n_flagged` the caller expects.

    Raises
    ------
    RuntimeError
        If `db.validate()` reports any problems, or `n_flagged` does not
        equal `expect_flagged` when one is given.
    """
    problems = db.validate()
    if problems:
        raise RuntimeError(f"validate(): {problems}")
    check_row_counts(db, counts)
    check_site_event_matches_records(db)
    check_flags(db, n_flagged)
    # cs_nshm_2022: the Slurm job knows how many gap sites the campaigns leave,
    # counted from their station lists before the build.
    if expect_flagged is not None and n_flagged != expect_flagged:
        raise RuntimeError(f"{n_flagged} site_event rows carry the pilot's distances, expected {expect_flagged}")


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

    At least `min_pilot` of the sampled realisations are pilot ones, where there
    are any (cs_nshm_2022: so both campaigns are always sampled).

    Parameters
    ----------
    db_path : Path
        Path to the finished (or `.partial`) database.
    manifest : dict of str to ManifestRow
        Every ingested realisation's manifest row, by rel_id.
    content : Content, optional
        Which components, periods and spectra the database holds (default DEFAULT_CONTENT).
    grids : Grids, optional
        The database's pSA periods and FAS frequencies (default DEFAULT_GRIDS).
    n_realisations : int, optional
        How many realisations to sample (default 25).
    per_realisation : int, optional
        How many records to sample per realisation (default 40).
    min_pilot : int, optional
        The minimum number of sampled realisations that must be pilot ones,
        where any exist (default 5).
    seed : int, optional
        Random seed for reproducible sampling (default 0).

    Returns
    -------
    int
        How many records were checked.

    Raises
    ------
    AssertionError
        If a sampled record's pSA, scalar IM or FAS values differ from the
        h5, are present for a component that should have none, or are not
        NULL where the h5 has none.
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


# cs_nshm_2022: some realisations reached physically implausible PGA near the
# fault, so the build log lists the largest for a human to judge before the
# database is shared. Another dataset can drop this, but it costs little to keep.
def peak_line(row: ManifestRow, peak: dict[str, tuple[float, str]]) -> str:
    """One realisation's largest PGA and PGV (geom), with where they are.

    Parameters
    ----------
    row : ManifestRow
        The realisation, for its label.
    peak : dict of str to tuple of (float, str)
        `PGA` and `PGV` mapped to (largest value, its station).

    Returns
    -------
    str
        A one-line summary naming the realisation, its peak PGA and its peak PGV.
    """
    (pga, pga_site), (pgv, pgv_site) = peak["PGA"], peak["PGV"]
    return f"{row.label}: PGA {pga:.3f} g at {pga_site}; PGV {pgv:.1f} cm/s at {pgv_site}"


def print_peaks(peaks: Sequence[tuple[ManifestRow, dict[str, tuple[float, str]]]]) -> None:
    """The realisations with the largest PGA, for a human to judge before the database is shared.

    Parameters
    ----------
    peaks : Sequence of (ManifestRow, dict of str to tuple of (float, str))
        Each realisation with its `PGA`/`PGV` peaks, as from `read_event`.
    """
    ranked = sorted(peaks, key=lambda item: item[1]["PGA"][0], reverse=True)
    print(f"top {min(N_TOP_PEAKS, len(ranked))} realisations by max PGA (geom):", flush=True)
    for row, peak in ranked[:N_TOP_PEAKS]:
        print(f"  {peak_line(row, peak)}", flush=True)


# ---- 6. Command line -----------------------------------------------------------------


def main() -> None:
    """Parse the command line, build the database, and print its row counts."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="build manifest CSV from build_imdb_manifest.py")
    parser.add_argument("stations_input", type=Path, help="the campaign's stations_input.ll (lon lat name)")
    parser.add_argument("nshm_csv", type=Path, help="campaign_nshm_ruptures.csv from export_nshm_rupture_attributes.py")
    parser.add_argument("out", type=Path, help="output .duckdb; must not exist")
    # cs_nshm_2022: provenance for db_meta, supplied by the Slurm job.
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
    # cs_nshm_2022: the number of pilot-only sites in shared events, from the Slurm job.
    parser.add_argument(
        "--expect-flagged",
        type=int,
        help="fail unless exactly this many site_event rows carry the pilot's distances",
    )
    # DuckDB sizes its memory from the node's RAM, not from the Slurm job's
    # allocation, and a job that outgrows its allocation is killed.
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
