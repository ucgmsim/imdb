"""Tests for scripts/cs_nshm_2022_ingest.py, on a synthetic two-campaign dataset.

imdb's own CI installs neither h5py nor source-modelling, so this module skips
there. Run it inside the imdb_tools image:
    apptainer exec imdb_tools_<date>.sif python3 -m pytest -q scripts/tests
"""

import csv
import hashlib
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
from shapely import unary_union, wkt  # noqa: E402
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
FAS_FREQUENCIES = (0.1, 1.0, 10.0)
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
# vs30, z1pt0, z2pt5 per station: global, like the real campaigns'.
SITE_VALUES = {
    "AMBC": (300.0, 0.3, 1.3),
    "2O8aUSL": (400.0, 0.2, 1.2),
    "2O8bSSL": (500.0, 0.15, 1.1),
    "2TfEMSL": (600.0, 0.1, 1.0),
    "2TfFKSL": (700.0, 0.05, 0.9),
}
MAIN_SHA256 = "3fe692cb9b22c769b6a9baa6a526b4eed989bd61ec054a344f0e74d82855bc4e"


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
    seed: int,
    domain_shift: float = 0.0,
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
        "seeds": {"hf_seed": seed, "genslip_seed": 1000 + seed},
        "domain": {
            "domain": [
                {"latitude": -43.0 + domain_shift, "longitude": 172.0 + domain_shift},
                {"latitude": -43.0 + domain_shift, "longitude": 173.0 + domain_shift},
                {"latitude": -44.0 + domain_shift, "longitude": 173.0 + domain_shift},
                {"latitude": -44.0 + domain_shift, "longitude": 172.0 + domain_shift},
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
                group["frequency"] = np.array(FAS_FREQUENCIES)
                for c in components:
                    group[c] = rng.uniform(size=(n, len(FAS_FREQUENCIES)))
            else:
                for c in components:
                    group[c] = rng.uniform(0.01, 5.0, n)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Campaign:
    manifest: Path
    files: dict[str, tuple[Path, Path]]
    """original_id -> (intensity_measures.h5, realisation.json)"""
    stations_input: Path
    nshm_csv: Path
    out: Path

    def h5(self, original_id: str) -> Path:
        return self.files[original_id][0]

    def realisation_json(self, original_id: str) -> Path:
        return self.files[original_id][1]

    def realisation(self, original_id: str) -> dict:
        return json.loads(self.realisation_json(original_id).read_text())

    def build(self, content=None, **kwargs) -> dict[str, int]:
        return ingest.build_database(
            self.manifest,
            self.stations_input,
            self.nshm_csv,
            self.out,
            harvested_at="2026-09-23T07:35:00Z",
            script_commit="abc123",
            content=content or ingest.Content(),
            **kwargs,
        )


# (original_id, rel_id, pilot, stations) in manifest order
REALISATIONS = [
    ("3", "3_R1", True, ["AMBC", "2O8aUSL"]),
    ("161984/R2", "161984_R1", False, ["AMBC", "2O8aUSL", "2O8bSSL", "2TfEMSL"]),
    ("161984/R3", "161984_R2", False, ["AMBC", "2O8aUSL", "2O8bSSL", "2TfEMSL"]),
    ("288271/R2", "288271_R1", False, ["2O8bSSL", "2TfEMSL", "2TfFKSL"]),
    ("288271", "288271_R2", True, ["2TfEMSL", "2TfFKSL", "AMBC", "2O8aUSL"]),
]
STATIONS = {original_id: stations for original_id, _, _, stations in REALISATIONS}
NSHM_MAGNITUDE = {"3": 6.9, "161984": 7.480455658305042, "288271": 7.13806256610197}


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    """Three events.

    - 161984: two main realisations on two faults.
    - 288271: one main realisation and a pilot whose fault's bottom edge sits
      elsewhere, whose domain is shifted, and whose stations only partly overlap.
    - 3: a pilot realisation alone.
    """
    two_faults = {"Clarence": _plane(-43.5, 172.5, 45.0), "Kekerengu": _plane(-43.45, 172.58, 50.0)}
    realisations = {
        "3": _realisation("3", {"Acton": _plane(-43.2, 172.2, 10.0)}, {"Acton": 6.8}, {"Acton": None}, (0.4, 0.5), 90.0, 1),
        "161984/R2": _realisation(
            "161984", two_faults, {"Clarence": 7.11, "Kekerengu": 6.59},
            {"Kekerengu": "Clarence", "Clarence": None}, (0.17, 0.19), 161.7, 2,
        ),
        "161984/R3": _realisation(
            "161984", two_faults, {"Clarence": 6.54, "Kekerengu": 6.02},
            {"Clarence": "Kekerengu", "Kekerengu": None}, (0.59, 0.92), 156.5, 3,
        ),
        "288271/R2": _realisation(
            "288271", {"Monowai": _plane(-43.6, 172.4, 30.0)}, {"Monowai": 7.32}, {"Monowai": None}, (0.5, 0.6), 120.0, 4,
        ),
        "288271": _realisation(
            "288271", {"Monowai": _plane(-43.6, 172.4, 33.0)}, {"Monowai": 7.14}, {"Monowai": None}, (0.1, 0.9), 110.0, 5,
            domain_shift=0.2,
        ),
    }  # fmt: skip
    rng = np.random.default_rng(1)
    distances = {
        "3": rng.uniform(0, 80, (2, 4)),
        "161984": rng.uniform(0, 80, (4, 4)),
        "288271/R2": rng.uniform(0, 80, (3, 4)),
        "288271": rng.uniform(0, 80, (4, 4)),
    }
    files, rows = {}, []
    for seed, (original_id, rel, pilot, stations) in enumerate(REALISATIONS):
        base = tmp_path / ("pilot" if pilot else "main") / original_id
        h5_path, json_path = base / "intensity_measures.h5", base / "realisation.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(realisations[original_id]))
        rupture_distances = distances["161984"] if original_id.startswith("161984") else distances[original_id]
        _write_h5(h5_path, stations, realisations[original_id], rupture_distances, seed)
        files[original_id] = (h5_path, json_path)
        event_id, n = rel.split("_R")
        rows.append(
            {
                "rel_id": rel,
                "event_id": event_id,
                "realisation": n,
                "pilot": "true" if pilot else "false",
                "campaign": "run2" if pilot else "run3",
                "original_id": original_id,
                "im_path": str(h5_path),
                "realisation_path": str(json_path),
            }
        )
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    stations_input = tmp_path / "stations_input.ll"
    stations_input.write_text("".join(f"{lon} {lat} {s}\n" for s, (lon, lat) in STATIONS_INPUT.items()))
    nshm_csv = tmp_path / "campaign_nshm_ruptures.csv"
    pd.DataFrame(
        {
            "rupture_id": ["3", "607", "161984", "288271"],
            "nshmdb_rupture_id": [3001, 40083, 201460, 327747],
            "magnitude_boldm": [NSHM_MAGNITUDE["3"], 7.47, NSHM_MAGNITUDE["161984"], NSHM_MAGNITUDE["288271"]],
            "area_km2": [300.0, 1872.5, 1907.5, 867.1],
            "length_km": [20.0, 59.9, 80.2, 34.6],
            "annual_rate": [1e-05, 6.1e-06, 1.86e-04, 7.97e-05],
            "nshmdb_file": "nshmdb_v2026.08.3.db",
            "nshmdb_sha256": MAIN_SHA256,
        }
    ).to_csv(nshm_csv, index=False)
    return Campaign(manifest, files, stations_input, nshm_csv, tmp_path / "out.duckdb")


def _manifest_rows(campaign: Campaign) -> list[dict[str, str]]:
    with campaign.manifest.open(newline="") as f:
        return list(csv.DictReader(f))


# ---- helpers -------------------------------------------------------------------


def test_select_period_indices_picks_the_exact_columns():
    columns = ingest.select_period_indices(np.array(VALID_PERIODS))
    assert tuple(np.array(VALID_PERIODS)[columns]) == ingest.PSA_PERIODS
    assert len(ingest.PSA_PERIODS) == 25


def test_select_period_indices_rejects_a_missing_period():
    without = np.array([p for p in VALID_PERIODS if p != 0.3])
    with pytest.raises(ValueError, match="0.3 s matches 0"):
        ingest.select_period_indices(without)


def test_is_real_station():
    assert ingest.is_real_station("AMBC")
    assert ingest.is_real_station("BFZ")
    assert not ingest.is_real_station("2O8aUSL")


def test_read_manifest_returns_ingest_order(campaign: Campaign):
    rows = _manifest_rows(campaign)
    _write_manifest(campaign.manifest, rows[::-1])
    parsed = ingest.read_manifest(campaign.manifest)
    assert [row.rel_id for row in parsed] == ["3_R1", "161984_R1", "161984_R2", "288271_R1", "288271_R2"]
    assert [row.pilot for row in parsed] == [True, False, False, False, True]
    assert parsed[1].original_id == "161984/R2"
    assert parsed[1].im_path == campaign.h5("161984/R2")


def _set(rows: list[dict[str, str]], rel: str, **changes: str) -> list[dict[str, str]]:
    return [{**row, **changes} if row["rel_id"] == rel else row for row in rows]


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda rows: _set(rows, "161984_R2", rel_id="161984_R3", realisation="3"), r"realisations \[1, 3\]"),
        (lambda rows: _set(_set(rows, "288271_R1", pilot="true", campaign="run2"), "288271_R2", pilot="false", campaign="run3"),
         "the pilot is R1"),
        (lambda rows: _set(rows, "288271_R1", pilot="true", campaign="run2"), "2 pilot realisations"),
        (lambda rows: [*rows, rows[0]], "3_R1: listed twice"),
        (lambda rows: _set(rows, "3_R1", rel_id="3_R9"), "3_R9: expected 3_R1"),
        (lambda rows: _set(rows, "3_R1", event_id="03", rel_id="03_R1"), "not an unpadded integer"),
        (lambda rows: _set(rows, "3_R1", pilot="yes"), "pilot must be true or false"),
        (lambda rows: [{k: v for k, v in row.items() if k != "original_id"} for row in rows], "missing columns"),
        (lambda rows: _set(rows, "3_R1", pilot="false"), r"campaign \['run2'\] has both pilot and non-pilot rows"),
    ],
    ids=["gap", "pilot-not-last", "two-pilots", "duplicate", "rel-id-mismatch", "padded-event", "bad-pilot", "no-column",
         "pilot-flag-vs-campaign"],
)  # fmt: skip
def test_bad_manifests_are_refused(campaign: Campaign, edit, match):
    _write_manifest(campaign.manifest, edit(_manifest_rows(campaign)))
    with pytest.raises(ValueError, match=match):
        ingest.read_manifest(campaign.manifest)


def test_an_empty_manifest_is_refused(campaign: Campaign):
    campaign.manifest.write_text(campaign.manifest.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="lists no realisations"):
        ingest.read_manifest(campaign.manifest)


def test_missing_files_are_found_before_the_build_starts(campaign: Campaign):
    campaign.realisation_json("161984/R3").unlink()
    with pytest.raises(FileNotFoundError, match=r"161984_R2 \(161984/R3\)"):
        ingest.read_manifest(campaign.manifest)


# ---- the whole build -------------------------------------------------------------


def test_row_counts_and_meta(campaign: Campaign):
    counts = campaign.build()
    assert counts == {
        "events": 3,
        "realisations": 5,
        "sites": 5,
        "site_event": 11,  # 2 + 4 + (3 main + 2 only the pilot has)
        "records": 34,
        "psa_ims": 34,
        "scalars_ims": 34,
        "fas_ims": 0,
    }
    assert campaign.out.exists()
    assert not campaign.out.with_name("out.duckdb.partial").exists()
    with IMDB(campaign.out) as db:
        meta = db.db_meta
        assert db.validate() == []
    assert meta["dataset_id"] == "cs_nshm_2022"
    assert meta["components"] == "geom,rotd50"
    assert meta["n_periods"] == "25"
    assert meta["n_frequencies"] == "0"
    assert meta["psa_period_selection"].startswith("25 of im-calc's 111 periods")
    assert meta["fas"] == "not included"
    assert meta["n_realisations"] == "5"
    assert meta["n_pilot_realisations"] == "2"
    assert meta["im_calc_harvested_at"] == "2026-09-23T07:35:00Z"
    assert meta["build_manifest_md5"] == hashlib.md5(campaign.manifest.read_bytes()).hexdigest()
    assert meta["ingest_script_commit"] == "abc123"
    assert meta["magnitude_convention"].startswith("BoldM")
    for key in ("description", "realisation_numbering", "pilot_realisations", "distances", "nshm_fault_database"):
        assert meta[key]
    assert "nshmdb" not in meta
    assert meta["source"] == "cs_nshm_2022 physics-based simulations; see description."
    for key, value in meta.items():  # nothing researcher-facing names our internals
        for word in ("run2", "run3", "/gpfs", "BSC", "envelope"):
            assert word not in value, f"db_meta {key} mentions {word}"


def test_events(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        events = db.get_events()
    for event_id, magnitude in NSHM_MAGNITUDE.items():
        assert events.loc[event_id, "magnitude"] == pytest.approx(magnitude, abs=1e-6)
    assert (events["tect_type"] == "ACTIVE_SHALLOW").all()
    assert pd.isna(events.loc["161984", "dip"])  # multi-fault: no single dip
    assert events.loc["288271", "dip"] == pytest.approx(60.0, abs=1e-3)
    metadata = {e: json.loads(events.loc[e, "metadata"]) for e in NSHM_MAGNITUDE}
    assert metadata["161984"]["nshmdb_rupture_id"] == 201460
    assert metadata["161984"]["annual_rate"] == 1.86e-04
    assert set(metadata["161984"]["segments"]) == {"Clarence", "Kekerengu"}
    assert "nshm_rupture_name" not in metadata["161984"]  # it only ever held "Rupture <id>"
    assert {e: m["fault_geometry_release"] for e, m in metadata.items()} == {
        "3": "pre-v2026.08",
        "161984": "v2026.08.3",
        "288271": "v2026.08.3",
    }


def test_event_geometry_comes_from_the_main_realisation(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        events = db.get_events()

    def geometry_of(original_id: str) -> str:
        return ingest.source_wkt(ingest.fault_geometries(campaign.realisation(original_id)))

    assert events.loc["288271", "source_wkt"] == geometry_of("288271/R2")
    assert events.loc["288271", "source_wkt"] != geometry_of("288271")
    assert events.loc["3", "source_wkt"] == geometry_of("3")


def test_domain_is_the_union_of_the_realisations_domains(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        events = db.get_events()
    main, pilot = (ingest.domain_polygon(campaign.realisation(o)) for o in ("288271/R2", "288271"))
    stored = wkt.loads(events.loc["288271", "domain_wkt"])
    assert stored.equals(unary_union([main, pilot]))
    assert stored.area > main.area
    assert wkt.loads(events.loc["161984", "domain_wkt"]).equals(ingest.domain_polygon(campaign.realisation("161984/R2")))


def test_realisations_match_their_own_realisation_json(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        realisations = db.get_realisations()
    for original_id, rel, pilot, _ in REALISATIONS:
        realisation = campaign.realisation(original_id)
        row = realisations.loc[rel]
        assert row["magnitude"] == pytest.approx(_boldm_total(realisation), abs=1e-5)
        # the pilot's hypocentre sits on the pilot's own fault, not the event's
        assert (row["hypo_lat"], row["hypo_lon"], row["hypo_depth"]) == pytest.approx(
            _hypocentre(realisation), abs=1e-4
        )
        metadata = json.loads(row["metadata"])
        assert metadata["pilot"] is pilot
        assert metadata["seeds"] == realisation["seeds"]
        assert metadata["duration_s"] == realisation["domain"]["duration"]
    # 161984_R2 starts on the other fault, so its causality tree is its own
    assert json.loads(realisations.loc["161984_R2", "metadata"])["rupture_causality_tree"]["Kekerengu"] is None


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


def _h5_distances(path: Path) -> pd.DataFrame:
    with h5py.File(path) as h5:
        stations = [s.decode() for s in h5["pSA/station"][:]]
        return pd.DataFrame({key: h5[f"pSA/{key}"][:] for key in ingest.DISTANCE_FIELDS}, index=stations)


def test_site_event_prefers_the_main_campaign_and_flags_the_pilots(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out) as db:
        site_event = db.get_site_event()
    by_event = {e: df.set_index("site_id") for e, df in site_event.groupby("event_id")}
    main, pilot = _h5_distances(campaign.h5("288271/R2")), _h5_distances(campaign.h5("288271"))
    shared = by_event["288271"]
    assert sorted(shared.index) == sorted(STATIONS_INPUT)
    for site in STATIONS["288271/R2"]:  # the main realisation's stations, 2TfEMSL included
        assert shared.loc[site, list(ingest.DISTANCE_FIELDS)].to_numpy() == pytest.approx(main.loc[site].to_numpy())
        assert pd.isna(shared.loc[site, "metadata"])
    for site in ("AMBC", "2O8aUSL"):  # only the pilot has these
        assert shared.loc[site, list(ingest.DISTANCE_FIELDS)].to_numpy() == pytest.approx(pilot.loc[site].to_numpy())
        assert json.loads(shared.loc[site, "metadata"]) == {"fault_geometry": "pilot"}
    # the pilot-only event keeps the pilot's distances, unflagged
    alone = _h5_distances(campaign.h5("3"))
    for site in STATIONS["3"]:
        assert by_event["3"].loc[site, list(ingest.DISTANCE_FIELDS)].to_numpy() == pytest.approx(alone.loc[site].to_numpy())
    assert by_event["3"]["metadata"].isna().all()
    assert by_event["161984"]["metadata"].isna().all()


def test_ims_are_the_h5_values(campaign: Campaign):
    campaign.build()
    for original_id, rel in (("161984/R3", "161984_R2"), ("288271", "288271_R2")):
        with h5py.File(campaign.h5(original_id)) as h5:
            columns = ingest.select_period_indices(h5["pSA/period"][:])
            station = STATIONS[original_id].index("2TfEMSL")
            psa_rotd50 = h5["pSA/rotd50"][station, columns].astype(np.float32)
            cav_geom = np.float32(h5["CAV/geom"][station])
            pga_rotd50 = np.float32(h5["PGA/rotd50"][station])
        with IMDB(campaign.out) as db:
            rotd50 = db.get_records(rel_ids=[rel], site_ids=["2TfEMSL"], component="rotd50").index
            geom = db.get_records(rel_ids=[rel], site_ids=["2TfEMSL"], component="geom").index
            psa = db.get_psa(record_int_ids=list(rotd50))
            scalars = db.get_scalars(record_int_ids=[*rotd50, *geom])
        np.testing.assert_array_equal(psa.iloc[0].to_numpy(dtype=np.float32), psa_rotd50)
        assert list(psa.columns) == [str(p) for p in ingest.PSA_PERIODS]
        assert np.float32(scalars.loc[rotd50[0], "PGA"]) == pga_rotd50
        assert pd.isna(scalars.loc[rotd50[0], "CAV"])  # undefined on rotd
        assert np.float32(scalars.loc[geom[0], "CAV"]) == cav_geom


def test_peaks_are_logged(campaign: Campaign, capsys: pytest.CaptureFixture):
    campaign.build()
    out = capsys.readouterr().out
    with h5py.File(campaign.h5("288271")) as h5:
        pga = h5["PGA/geom"][:]
        site = h5["PGA/station"][int(np.argmax(pga))].decode()
    assert f"peak 288271_R2 (288271): PGA {pga.max():.3f} g at {site}" in out
    assert "top 5 realisations by max PGA (geom):" in out


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


def _edit_json(path: Path, edit) -> None:
    realisation = json.loads(path.read_text())
    edit(realisation)
    path.write_text(json.dumps(realisation))


@pytest.mark.parametrize(
    "corrupt, error, match",
    [
        (lambda c: _edit_h5(c.h5("288271/R2"), "geom", (0, KEPT_COLUMN), np.nan, ("pSA",)), ValueError, "pSA/geom has 1 non-finite"),
        (lambda c: _edit_h5(c.h5("288271/R2"), "vs30", 1, 999.0), ValueError, "earlier realisation"),
        (lambda c: _edit_h5(c.h5("288271"), "vs30", 0, 999.0), ValueError, r"288271_R2 \(288271\): 2TfEMSL"),
        (lambda c: _edit_h5(c.h5("161984/R3"), "rrup", 2, 1.5), ValueError, r"rrup differs from 161984_R1 \(161984/R2\)"),
        (lambda c: _set_attr(c.h5("161984/R3"), "magnitude", 0.01), ValueError, "magnitude from realisation.json"),
        (lambda c: _set_attr(c.h5("288271"), "hypo_lat", 0.001), ValueError, "hypo_lat from realisation.json"),
        (lambda c: c.stations_input.write_text(c.stations_input.read_text().replace("2TfFKSL", "XXXXXXX")),
         ValueError, "missing from stations_input.ll"),
        (lambda c: c.h5("288271/R2").unlink(), FileNotFoundError, r"288271_R1 \(288271/R2\)"),
        (lambda c: _edit_json(c.realisation_json("3"), lambda r: r.update(seeds=c.realisation("161984/R3")["seeds"])),
         ValueError, "the same seeds as 3_R1"),
    ],
    ids=["nan", "site-mismatch", "pilot-site-mismatch", "distance-mismatch", "magnitude", "hypocentre",
         "unknown-station", "no-h5", "same-seeds"],
)  # fmt: skip
def test_bad_input_stops_the_build(campaign: Campaign, corrupt, error, match):
    corrupt(campaign)
    with pytest.raises(error, match=match):
        campaign.build()
    assert not campaign.out.exists()


def test_files_of_another_event_stop_the_build(campaign: Campaign):
    rows = _manifest_rows(campaign)
    three = next(row for row in rows if row["rel_id"] == "3_R1")
    other = next(row for row in rows if row["rel_id"] == "288271_R2")
    for key in ("im_path", "realisation_path"):
        three[key], other[key] = other[key], three[key]
    _write_manifest(campaign.manifest, rows)
    with pytest.raises(ValueError, match=r"3_R1 \(3\): realisation.json describes 'Rupture 288271', not event 3"):
        campaign.build()


def test_main_realisations_must_share_their_stations(campaign: Campaign):
    _write_h5(campaign.h5("161984/R3"), STATIONS["161984/R3"][:3], campaign.realisation("161984/R3"), np.zeros((3, 4)), 9)
    with pytest.raises(ValueError, match=r"161984_R2 \(161984/R3\): station list differs from 161984_R1"):
        campaign.build()


def test_expected_flag_count(campaign: Campaign):
    with pytest.raises(RuntimeError, match="2 site_event rows carry the pilot's distances, expected 3"):
        campaign.build(expect_flagged=3)
    campaign.out.with_name("out.duckdb.partial").unlink()
    assert campaign.build(expect_flagged=2)["site_event"] == 11


def test_the_table_checks_catch_a_tampered_database(campaign: Campaign):
    campaign.build()
    with IMDB(campaign.out, read_only=False) as db:
        ingest.check_site_event_matches_records(db)
        ingest.check_flags(db, 2)
        with pytest.raises(RuntimeError, match="2 flagged rows"):
            ingest.check_flags(db, 5)
        db.con.raw_sql("DELETE FROM site_event WHERE metadata IS NOT NULL")
        with pytest.raises(RuntimeError, match="0 rows with no records, 2 record pairs with no row"):
            ingest.check_site_event_matches_records(db)


def test_release_memtables(tmp_path: Path):
    db = IMDB.create(tmp_path / "m.duckdb", periods=[0.1], components=("geom",))
    db.add_events(pd.DataFrame({"event_id": ["1"]}))
    memtables = "SELECT count(*) FROM duckdb_views() WHERE temporary AND view_name LIKE 'ibis_pandas_memtable_%'"
    assert db.con.raw_sql(memtables).fetchone()[0] > 0
    ingest.release_memtables(db)
    assert db.con.raw_sql(memtables).fetchone()[0] == 0
    db.add_realisations(pd.DataFrame({"rel_id": ["1_R1"], "event_id": ["1"]}))  # still writable
    db.close()


def test_the_pilots_distances_are_not_compared_with_the_main_campaigns(campaign: Campaign):
    main, pilot = _h5_distances(campaign.h5("288271/R2")), _h5_distances(campaign.h5("288271"))
    assert not np.allclose(main.loc["2TfEMSL"], pilot.loc["2TfEMSL"])
    campaign.build()


def test_unknown_tect_type_stops_the_build(campaign: Campaign):
    _edit_json(campaign.realisation_json("288271/R2"), lambda r: r["empirical"].update(tect_type="subduction_mystery"))
    with pytest.raises(ValueError, match="tect_type"):
        campaign.build()


def test_rupture_missing_from_the_nshm_csv(campaign: Campaign):
    nshm = pd.read_csv(campaign.nshm_csv)
    nshm[nshm["rupture_id"] != 288271].to_csv(campaign.nshm_csv, index=False)
    with pytest.raises(ValueError, match=r"missing from .*\['288271'\]"):
        campaign.build()


def test_nshm_attributes_from_another_release_are_refused(campaign: Campaign):
    nshm = pd.read_csv(campaign.nshm_csv)
    nshm.loc[0, "nshmdb_sha256"] = "00e25648"
    nshm.to_csv(campaign.nshm_csv, index=False)
    with pytest.raises(ValueError, match="must all come from v2026.08.3"):
        campaign.build()


def test_nan_in_a_dropped_period_is_ignored(campaign: Campaign):
    _edit_h5(campaign.h5("288271/R2"), "geom", (0, DROPPED_COLUMN), np.nan, ("pSA",))
    assert campaign.build()["records"] == 34


# ---- content options -------------------------------------------------------------

EVERYTHING = ingest.Content(
    components=("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100", "eas"),
    all_periods=True,
    fas=True,
)


def test_default_content():
    content = ingest.Content()
    assert content.components == ("geom", "rotd50")
    assert content.psa_components == ("geom", "rotd50")
    assert content.fas_components == ()
    assert EVERYTHING.psa_components == ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")
    assert EVERYTHING.fas_components == ("000", "090", "ver", "geom", "eas")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"components": ("geom", "eas")}, "eas has only FAS"),
        ({"components": ("geom", "rotd90")}, "unknown components"),
        ({"components": ("geom", "geom")}, "listed twice"),
        ({"components": ()}, "no components"),
        ({"components": ("rotd50",), "fas": True}, "--fas needs a component that has FAS"),
    ],
)
def test_bad_content_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ingest.Content(**kwargs)


def test_everything_on(campaign: Campaign):
    n_stations = sum(len(stations) for stations in STATIONS.values())  # 17
    counts = campaign.build(EVERYTHING)
    assert counts == {
        "events": 3,
        "realisations": 5,
        "sites": 5,
        "site_event": 11,
        "records": 8 * n_stations,
        "psa_ims": 7 * n_stations,
        "scalars_ims": 7 * n_stations,
        "fas_ims": 5 * n_stations,
    }
    with IMDB(campaign.out) as db:
        meta = db.db_meta
        assert db.validate() == []
        rel, site = "288271_R2", "2TfEMSL"
        ids = {c: db.get_records(rel_ids=[rel], site_ids=[site], component=c).index[0] for c in EVERYTHING.components}
        psa = db.get_psa(record_int_ids=list(ids.values()))
        fas = db.get_fas(record_int_ids=list(ids.values()))
        scalars = db.get_scalars(record_int_ids=list(ids.values()))
    assert meta["components"] == "000,090,ver,geom,rotd0,rotd50,rotd100,eas"
    assert meta["n_periods"] == "111"
    assert meta["n_frequencies"] == "3"
    assert meta["psa_period_selection"] == "all 111 of im-calc's periods, 0.01-20 s."
    assert meta["fas"].startswith("all 3 of im-calc's FAS frequencies, 0.1-10 Hz, for 000, 090, ver, geom, eas")
    with h5py.File(campaign.h5("288271")) as h5:
        i = STATIONS["288271"].index(site)
        np.testing.assert_array_equal(psa.loc[ids["090"]].to_numpy(dtype=np.float32), h5["pSA/090"][i].astype(np.float32))
        np.testing.assert_array_equal(fas.loc[ids["000"]].to_numpy(dtype=np.float32), h5["FAS/000"][i].astype(np.float32))
        np.testing.assert_array_equal(fas.loc[ids["eas"]].to_numpy(dtype=np.float32), h5["FAS/eas"][i].astype(np.float32))
        assert np.float32(scalars.loc[ids["ver"], "CAV"]) == np.float32(h5["CAV/ver"][i])
    assert ids["rotd50"] not in fas.index  # no rotd FAS
    assert ids["eas"] not in psa.index and ids["eas"] not in scalars.index  # eas is FAS only
    assert list(psa.columns) == [str(p) for p in VALID_PERIODS]


def test_a_different_fas_grid_stops_the_build(campaign: Campaign):
    with h5py.File(campaign.h5("288271"), "r+") as h5:
        h5["FAS/frequency"][...] = np.array([0.1, 1.0, 20.0])
    with pytest.raises(ValueError, match="FAS frequency grid differs"):
        campaign.build(EVERYTHING)


def test_a_different_period_grid_stops_an_all_periods_build(campaign: Campaign):
    with h5py.File(campaign.h5("288271"), "r+") as h5:
        h5["pSA/period"][0] = 0.015
    with pytest.raises(ValueError, match="0.01 s matches 0"):
        campaign.build(ingest.Content(all_periods=True))


def test_eas_alone(campaign: Campaign):
    n_stations = sum(len(stations) for stations in STATIONS.values())
    counts = campaign.build(ingest.Content(components=("eas",), fas=True))
    assert (counts["records"], counts["psa_ims"], counts["scalars_ims"], counts["fas_ims"]) == (n_stations, 0, 0, n_stations)
    with IMDB(campaign.out) as db:
        meta = db.db_meta
        assert db.validate() == []
    assert meta["n_periods"] == "0"
    assert meta["psa_period_selection"] == "not included"
    assert meta["fas"].startswith("all 3 of im-calc's FAS frequencies, 0.1-10 Hz, for eas")
