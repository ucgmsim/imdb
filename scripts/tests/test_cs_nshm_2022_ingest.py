"""Tests for scripts/cs_nshm_2022_ingest.py, on a synthetic campaign in run3's layout.

imdb's own CI installs neither h5py nor source-modelling, so this module skips
there. Run it inside the imdb_tools image:
    apptainer exec imdb_tools_<date>.sif python3 -m pytest -q scripts/tests
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("source_modelling")

from imdb import IMDB  # noqa: E402
from source_modelling import moment, sources  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "cs_nshm_2022_ingest.py"
_spec = importlib.util.spec_from_file_location("cs_nshm_2022_ingest", SCRIPT)
ingest = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ingest  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(ingest)

# im-calc's period grid, as in every run3 realisation.json `im.valid_periods`
VALID_PERIODS = (
    0.01, 0.02, 0.022, 0.025, 0.029, 0.03, 0.032, 0.035, 0.036, 0.04, 0.042, 0.044, 0.045,
    0.046, 0.048, 0.05, 0.055, 0.06, 0.065, 0.067, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,
    0.1, 0.11, 0.12, 0.13, 0.133, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.25,
    0.26, 0.28, 0.29, 0.3, 0.32, 0.34, 0.35, 0.36, 0.38, 0.4, 0.42, 0.44, 0.45, 0.46, 0.48,
    0.5, 0.55, 0.6, 0.65, 0.667, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6, 3.8, 4.0,
    4.2, 4.4, 4.6, 4.8, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 20.0,
)  # fmt: skip
ALL_COMPONENTS = ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")
AS_RECORDED = ("000", "090", "ver", "geom")
H5_GROUPS = {
    "pSA": ALL_COMPONENTS,
    "PGA": ALL_COMPONENTS,
    "PGV": ALL_COMPONENTS,
    "PGD": ALL_COMPONENTS,
    "CAV": AS_RECORDED,
    "AI": AS_RECORDED,
    "Ds575": AS_RECORDED,
    "Ds595": AS_RECORDED,
    "FAS": (*AS_RECORDED, "eas"),
}

# Canonical coordinates, deliberately not what the h5 files carry.
STATIONS_INPUT = {
    "AMBC": (172.1, -43.1),
    "2O8aUSL": (172.2, -43.2),
    "2O8bSSL": (172.3, -43.3),
    "2TfEMSL": (172.4, -43.4),
    "2TfFKSL": (172.5, -43.5),
}
# vs30, z1pt0, z2pt5 per station: global, like the real campaign's.
SITE_VALUES = {
    "AMBC": (300.0, 0.3, 1.3),
    "2O8aUSL": (400.0, 0.2, 1.2),
    "2O8bSSL": (500.0, 0.15, 1.1),
    "2TfEMSL": (600.0, 0.1, 1.0),
    "2TfFKSL": (700.0, 0.05, 0.9),
}


def _plane(lat: float, lon: float, strike: float) -> sources.Plane:
    return sources.Plane.from_centroid_strike_dip(
        np.array([lat, lon, 8.0]), 60.0, 10.0, 6.0, strike=strike
    )


def _realisation(
    name: str,
    faults: dict[str, sources.Plane],
    magnitudes: dict[str, float],
    tree: dict[str, str | None],
    hypocentre: tuple[float, float],
    duration: float,
) -> dict:
    return {
        "metadata": {"name": f"Rupture {name}"},
        "sources": {
            "source_geometries": {
                fault: {
                    "type": "fault",
                    "corners": [
                        {"latitude": c[0], "longitude": c[1], "depth": c[2]} for c in plane.corners
                    ],
                }
                for fault, plane in faults.items()
            }
        },
        "magnitudes": {"magnitudes": magnitudes},
        "rakes": {"rakes": dict.fromkeys(faults, 110.0)},
        "rupture_propagation": {
            "rupture_causality_tree": tree,
            "hypocentre": {"s": hypocentre[0], "d": hypocentre[1]},
        },
        "seeds": {"hf_seed": 7},
        "domain": {
            "domain": [
                {"latitude": -43.0, "longitude": 172.0},
                {"latitude": -43.0, "longitude": 173.0},
                {"latitude": -44.0, "longitude": 173.0},
                {"latitude": -44.0, "longitude": 172.0},
            ],
            "duration": duration,
            "depth": 40.0,
        },
        "empirical": {"tect_type": "active_shallow", "models": ["NSHM2022"]},
    }


def _boldm_total(realisation: dict) -> float:
    magnitudes = realisation["magnitudes"]["magnitudes"].values()
    return moment.moment_to_magnitude(
        sum(moment.magnitude_to_moment(m, bold_m=True) for m in magnitudes), bold_m=True
    )


def _hypocentre(realisation: dict) -> tuple[float, float, float]:
    tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    initial = next(name for name, parent in tree.items() if parent is None)
    entry = realisation["sources"]["source_geometries"][initial]
    corners = np.array([[c["latitude"], c["longitude"], c["depth"]] for c in entry["corners"]])
    fault = sources.Fault.from_corners(corners.reshape(-1, 4, 3))
    h = realisation["rupture_propagation"]["hypocentre"]
    lat, lon, depth_m = fault.fault_coordinates_to_wgs_depth_coordinates(np.array([h["s"], h["d"]]))
    return float(lat), float(lon), float(depth_m) / 1000


def _write_h5(path: Path, stations: list[str], realisation: dict, distances: np.ndarray, seed: int) -> None:
    """An intensity_measures.h5 in im-calc's layout, attributes matching `realisation`."""
    rng = np.random.default_rng(seed)
    n = len(stations)
    hypo_lat, hypo_lon, hypo_depth = _hypocentre(realisation)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in {
            "magnitude": _boldm_total(realisation),
            "hypo_lat": hypo_lat,
            "hypo_lon": hypo_lon,
            "hypo_depth": hypo_depth,
            "rake": 45.0,
        }.items():
            h5.attrs[key] = np.array([value])
        for group_name, components in H5_GROUPS.items():
            group = h5.create_group(group_name)
            group.create_dataset("station", data=stations, dtype=h5py.string_dtype())
            group["latitude"] = np.float32([STATIONS_INPUT[s][1] + 0.001 for s in stations])
            group["longitude"] = np.float32([STATIONS_INPUT[s][0] + 0.001 for s in stations])
            for i, key in enumerate(("vs30", "z1pt0", "z2pt5")):
                group[key] = np.array([SITE_VALUES[s][i] for s in stations])
            for i, key in enumerate(("rrup", "rjb", "rx", "ry")):
                group[key] = distances[:, i]
            group["epi"] = rng.uniform(1, 100, n)
            group["hyp"] = rng.uniform(1, 100, n)
            if group_name == "pSA":
                group["period"] = np.array(VALID_PERIODS)
                for c in components:
                    group[c] = rng.uniform(1e-3, 2.0, (n, len(VALID_PERIODS)))
            elif group_name == "FAS":
                group["frequency"] = np.array([0.1, 1.0, 10.0])
                for c in components:
                    group[c] = rng.uniform(size=(n, 3))
            else:
                for c in components:
                    group[c] = rng.uniform(0.01, 5.0, n)


@dataclass
class Campaign:
    share: Path
    events: Path
    stations_input: Path
    nshm_csv: Path
    realisations: list[tuple[str, int]]
    out: Path

    def h5(self, rel: str) -> Path:
        rupture, n = rel.split("/R")
        return self.share / rupture / f"R{n}" / "intensity_measures.h5"

    def realisation_json(self, rel: str) -> Path:
        rupture, n = rel.split("/R")
        return self.events / rupture / f"R{n}" / "realisation.json"

    def build(self) -> dict[str, int]:
        return ingest.build_database(
            self.share,
            self.events,
            self.stations_input,
            self.nshm_csv,
            self.realisations,
            self.out,
            source="pytest",
            script_commit="abc123",
        )


RUPTURE_STATIONS = {
    "161984": ["AMBC", "2O8aUSL", "2O8bSSL", "2TfEMSL"],
    "288271": ["2O8bSSL", "2TfEMSL", "2TfFKSL"],
}
NSHM_MAGNITUDE = {"161984": 7.480455658305042, "288271": 7.13806256610197}


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    """Two ruptures: 161984 (two faults, R2 and R3) and 288271 (one fault, R2)."""
    share, events = tmp_path / "share", tmp_path / "events"
    two_faults = {"Clarence": _plane(-43.5, 172.5, 45.0), "Kekerengu": _plane(-43.45, 172.58, 50.0)}
    one_fault = {"Monowai": _plane(-43.6, 172.4, 30.0)}
    realisations = {
        "161984/R2": _realisation(
            "161984", two_faults, {"Clarence": 7.11, "Kekerengu": 6.59},
            {"Kekerengu": "Clarence", "Clarence": None}, (0.17, 0.19), 161.7,
        ),
        "161984/R3": _realisation(
            "161984", two_faults, {"Clarence": 6.54, "Kekerengu": 6.02},
            {"Clarence": "Kekerengu", "Kekerengu": None}, (0.59, 0.92), 156.5,
        ),
        "288271/R2": _realisation(
            "288271", one_fault, {"Monowai": 7.32}, {"Monowai": None}, (0.5, 0.6), 120.0,
        ),
    }  # fmt: skip
    rng = np.random.default_rng(1)
    distances = {rupture: rng.uniform(0, 80, (len(s), 4)) for rupture, s in RUPTURE_STATIONS.items()}
    for seed, (rel, realisation) in enumerate(realisations.items()):
        rupture = rel.split("/")[0]
        path = events / rel / "realisation.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(realisation))
        _write_h5(share / rel / "intensity_measures.h5", RUPTURE_STATIONS[rupture], realisation, distances[rupture], seed)

    stations_input = tmp_path / "stations_input.ll"
    stations_input.write_text("".join(f"{lon} {lat} {s}\n" for s, (lon, lat) in STATIONS_INPUT.items()))
    nshm_csv = tmp_path / "campaign_nshm_ruptures.csv"
    pd.DataFrame(
        {
            "rupture_id": ["607", "161984", "288271"],
            "nshmdb_rupture_id": [40083, 201460, 327747],
            "magnitude_boldm": [7.47, NSHM_MAGNITUDE["161984"], NSHM_MAGNITUDE["288271"]],
            "area_km2": [1872.5, 1907.5, 867.1],
            "length_km": [59.9, 80.2, 34.6],
            "annual_rate": [6.1e-06, 1.86e-04, 7.97e-05],
            "nshmdb_file": "nshmdb_v2026.08.3.db",
            "nshmdb_sha256": "3fe692cb",
        }
    ).to_csv(nshm_csv, index=False)
    return Campaign(
        share, events, stations_input, nshm_csv,
        [("161984", 2), ("161984", 3), ("288271", 2)], tmp_path / "out.duckdb",
    )  # fmt: skip


# ---- helpers -------------------------------------------------------------------


def test_select_period_indices_picks_the_exact_columns():
    columns = ingest.select_period_indices(np.array(VALID_PERIODS))
    assert tuple(np.array(VALID_PERIODS)[columns]) == ingest.PSA_PERIODS
    assert len(ingest.PSA_PERIODS) == 25


def test_select_period_indices_rejects_a_missing_period():
    without = np.array([p for p in VALID_PERIODS if p != 0.3])
    with pytest.raises(ValueError, match="0.3 s matches 0"):
        ingest.select_period_indices(without)


def test_parse_realisation_ids():
    lines = ["161984/R3", "", "607/R2", "161984/R2", "607/R2"]
    assert ingest.parse_realisation_ids(lines) == [("607", 2), ("161984", 2), ("161984", 3)]
    with pytest.raises(ValueError, match="161984_R2"):
        ingest.parse_realisation_ids(["161984_R2"])


def test_is_real_station():
    assert ingest.is_real_station("AMBC")
    assert ingest.is_real_station("BFZ")
    assert not ingest.is_real_station("2O8aUSL")


# ---- the whole build -------------------------------------------------------------


def test_row_counts_and_meta(campaign: Campaign):
    counts = campaign.build()
    assert counts == {
        "events": 2,
        "realisations": 3,
        "sites": 5,
        "site_event": 7,
        "records": 22,
        "psa_ims": 22,
        "scalars_ims": 22,
    }
    assert campaign.out.exists()
    assert not campaign.out.with_name("out.duckdb.partial").exists()
    with IMDB(campaign.out) as db:
        meta = db.db_meta
        assert db.validate() == []
    assert meta["dataset_id"] == "cs_nshm_2022_run3"
    assert meta["components"] == "geom,rotd50"
    assert meta["n_periods"] == "25"
    assert meta["n_frequencies"] == "0"
    assert meta["magnitude_convention"].startswith("BoldM")
    assert meta["nshmdb"] == "nshmdb_v2026.08.3.db sha256 3fe692cb"
    assert meta["ingest_script_commit"] == "abc123"
    assert meta["source"] == "pytest"
    assert meta["n_realisations"] == "3"


def test_events(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        events = db.get_events()
    assert events.loc["161984", "magnitude"] == pytest.approx(NSHM_MAGNITUDE["161984"], abs=1e-6)
    assert events.loc["288271", "magnitude"] == pytest.approx(NSHM_MAGNITUDE["288271"], abs=1e-6)
    assert (events["tect_type"] == "ACTIVE_SHALLOW").all()
    assert pd.isna(events.loc["161984", "dip"])  # multi-fault: no single dip
    assert events.loc["288271", "dip"] == pytest.approx(60.0, abs=1e-3)
    assert events.loc["288271", "length"] == pytest.approx(10.0, abs=1e-3)
    metadata = json.loads(events.loc["161984", "metadata"])
    assert metadata["nshmdb_rupture_id"] == 201460
    assert metadata["annual_rate"] == 1.86e-04
    assert set(metadata["segments"]) == {"Clarence", "Kekerengu"}
    assert events.loc["161984", "source_wkt"].startswith("MULTIPOLYGON")


def test_realisations_are_boldm_and_match_their_h5(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        realisations = db.get_realisations()
    for rel in ("161984/R2", "161984/R3", "288271/R2"):
        realisation = json.loads(campaign.realisation_json(rel).read_text())
        row = realisations.loc[rel.replace("/", "_")]
        assert row["magnitude"] == pytest.approx(_boldm_total(realisation), abs=1e-5)
        assert (row["hypo_lat"], row["hypo_lon"], row["hypo_depth"]) == pytest.approx(
            _hypocentre(realisation), abs=1e-4
        )
        metadata = json.loads(row["metadata"])
        assert metadata["duration_s"] == realisation["domain"]["duration"]
    # R3 starts on the other fault, so its causality tree is its own
    assert json.loads(realisations.loc["161984_R3", "metadata"])["rupture_causality_tree"]["Kekerengu"] is None


def test_sites_use_canonical_coordinates(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        sites = db.get_sites()
    for site, (lon, lat) in STATIONS_INPUT.items():
        assert (sites.loc[site, "lat"], sites.loc[site, "lon"]) == pytest.approx((lat, lon), abs=1e-5)
        assert (sites.loc[site, "vs30"], sites.loc[site, "z1p0"], sites.loc[site, "z2p5"]) == pytest.approx(
            SITE_VALUES[site], abs=1e-5
        )
    assert sites["is_real"].to_dict() == {s: s == "AMBC" for s in STATIONS_INPUT}


def test_site_event_comes_from_the_h5(campaign: Campaign):
    campaign.build()
    with h5py.File(campaign.h5("288271/R2")) as h5:
        rrup = h5["pSA/rrup"][:]
    with IMDB(campaign.out) as db:
        site_event = db.get_site_event(event_ids=["288271"]).set_index("site_id")
    assert site_event.loc[RUPTURE_STATIONS["288271"], "rrup"].to_numpy() == pytest.approx(rrup, abs=1e-4)


def test_ims_are_the_h5_values(campaign: Campaign):
    campaign.build()
    with h5py.File(campaign.h5("161984/R3")) as h5:
        columns = ingest.select_period_indices(h5["pSA/period"][:])
        station = RUPTURE_STATIONS["161984"].index("2TfEMSL")
        psa_rotd50 = h5["pSA/rotd50"][station, columns].astype(np.float32)
        cav_geom = np.float32(h5["CAV/geom"][station])
        pga_rotd50 = np.float32(h5["PGA/rotd50"][station])
    with IMDB(campaign.out) as db:
        rotd50 = db.get_records(rel_ids=["161984_R3"], site_ids=["2TfEMSL"], component="rotd50").index
        geom = db.get_records(rel_ids=["161984_R3"], site_ids=["2TfEMSL"], component="geom").index
        psa = db.get_psa(record_int_ids=list(rotd50))
        scalars = db.get_scalars(record_int_ids=[*rotd50, *geom])
    np.testing.assert_array_equal(psa.iloc[0].to_numpy(dtype=np.float32), psa_rotd50)
    assert list(psa.columns) == [str(p) for p in ingest.PSA_PERIODS]
    assert np.float32(scalars.loc[rotd50[0], "PGA"]) == pga_rotd50
    assert pd.isna(scalars.loc[rotd50[0], "CAV"])  # undefined on rotd
    assert np.float32(scalars.loc[geom[0], "CAV"]) == cav_geom


# ---- refusals and consistency checks -------------------------------------------


def test_refuses_to_overwrite(campaign: Campaign):
    campaign.out.write_text("")
    with pytest.raises(FileExistsError):
        campaign.build()


def test_refuses_a_leftover_partial(campaign: Campaign):
    campaign.out.with_name("out.duckdb.partial").write_text("")
    with pytest.raises(FileExistsError, match="previous run died"):
        campaign.build()


KEPT_COLUMN = VALID_PERIODS.index(0.1)  # 26: the first period the database keeps
DROPPED_COLUMN = VALID_PERIODS.index(0.025)  # a period the database drops


def _edit_h5(path: Path, dataset: str, index, value, groups: tuple[str, ...] | None = None) -> None:
    """Overwrite one element of `dataset` in `groups` (default: every group that has it)."""
    with h5py.File(path, "r+") as h5:
        for name, group in h5.items():
            if dataset in group and (groups is None or name in groups):
                data = group[dataset][...]
                data[index] = value
                group[dataset][...] = data


def _set_attr(path: Path, key: str, delta: float) -> None:
    with h5py.File(path, "r+") as h5:
        h5.attrs[key] = h5.attrs[key] + delta


@pytest.mark.parametrize(
    "corrupt, error, match",
    [
        (lambda c: _edit_h5(c.h5("288271/R2"), "geom", (0, KEPT_COLUMN), np.nan, ("pSA",)), ValueError, "pSA/geom has 1 non-finite"),
        (lambda c: _edit_h5(c.h5("288271/R2"), "vs30", 1, 999.0), ValueError, "earlier rupture"),
        (lambda c: _edit_h5(c.h5("161984/R3"), "rrup", 2, 1.5), ValueError, "rrup differs from 161984_R2"),
        (lambda c: _set_attr(c.h5("161984/R3"), "magnitude", 0.01), ValueError, "magnitude from realisation.json"),
        (lambda c: _set_attr(c.h5("288271/R2"), "hypo_lat", 0.001), ValueError, "hypo_lat from realisation.json"),
        (lambda c: c.stations_input.write_text(c.stations_input.read_text().replace("2TfFKSL", "XXXXXXX")),
         ValueError, "missing from stations_input.ll"),
        (lambda c: c.h5("288271/R2").unlink(), FileNotFoundError, "288271_R2"),
    ],
    ids=["nan", "site-mismatch", "distance-mismatch", "magnitude", "hypocentre", "unknown-station", "no-h5"],
)  # fmt: skip
def test_bad_input_stops_the_build(campaign: Campaign, corrupt, error, match):
    corrupt(campaign)
    with pytest.raises(error, match=match):
        campaign.build()
    assert not campaign.out.exists()


def test_unknown_tect_type_stops_the_build(campaign: Campaign):
    path = campaign.realisation_json("288271/R2")
    realisation = json.loads(path.read_text())
    realisation["empirical"]["tect_type"] = "subduction_mystery"
    path.write_text(json.dumps(realisation))
    with pytest.raises(ValueError, match="tect_type"):
        campaign.build()


def test_rupture_missing_from_the_nshm_csv(campaign: Campaign):
    pd.read_csv(campaign.nshm_csv).iloc[:2].to_csv(campaign.nshm_csv, index=False)
    with pytest.raises(ValueError, match=r"missing from .*\['288271'\]"):
        campaign.build()


def test_nan_in_a_dropped_period_is_ignored(campaign: Campaign):
    _edit_h5(campaign.h5("288271/R2"), "geom", (0, DROPPED_COLUMN), np.nan, ("pSA",))
    assert campaign.build()["records"] == 22
