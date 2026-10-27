"""Tests for `IMDB.merge_from`: copying events between databases."""

import numpy as np
import pandas as pd
import pytest

from imdb import IMDB
from tests.conftest import COMPONENTS, FREQUENCIES, PERIODS


def _single_event_db(
    path, event_id="eventC", site_lat=-43.9, pga=0.25, extra_site=True
):
    """A database holding one simulated event at siteA (shared) and siteD (new)."""
    db = IMDB.create(
        path, periods=PERIODS, frequencies=FREQUENCIES, components=COMPONENTS
    )
    db.add_events(
        pd.DataFrame(
            {
                "event_id": [event_id],
                "magnitude": [5.5],
                "domain_wkt": ["POLYGON ((0 0, 1 0, 1 1, 0 0))"],
            }
        )
    )
    db.add_realisations(
        pd.DataFrame({"rel_id": [event_id], "event_id": [event_id], "rake": [90.0]})
    )
    site_ids = ["siteA", "siteD"] if extra_site else ["siteA"]
    db.add_sites(
        pd.DataFrame(
            {
                "site_id": site_ids,
                "lat": [site_lat, -44.0][: len(site_ids)],
                "lon": [172.0, 172.1][: len(site_ids)],
                "vs30": [400.0, 250.0][: len(site_ids)],
            }
        )
    )
    db.add_site_event(
        pd.DataFrame(
            {
                "site_id": site_ids,
                "event_id": [event_id] * len(site_ids),
                "rrup": np.arange(len(site_ids), dtype=float) + 5,
            }
        )
    )
    rows = [(site_id, component) for site_id in site_ids for component in COMPONENTS]
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": event_id,
                "site_id": [r[0] for r in rows],
                "component": [r[1] for r in rows],
                "kind": "simulated",
                "PGA": [pga] * len(rows),
            }
        ),
        pSA=np.full((len(rows), len(PERIODS)), pga),
        FAS=np.full((len(rows), len(FREQUENCIES)), pga / 10),
    )
    return db


def test_merge_adds_event_and_sites(db, tmp_path):
    """A new event, its new site and all its IMs are copied; shared sites are updated."""
    with _single_event_db(tmp_path / "src.duckdb") as src:
        merged = db.merge_from(src)

    assert merged == ["eventC"]
    assert db.validate() == []
    assert set(db.get_events().index) == {"eventA", "eventB", "eventC"}
    sites = db.get_sites()
    assert "siteD" in sites.index
    assert sites.loc["siteA", "lat"] == pytest.approx(-43.9)
    assert sites.loc["siteA", "vs30"] == pytest.approx(400.0)

    records = db.get_records(event_ids=["eventC"])
    assert len(records) == 4
    scalars = db.get_scalars(event_ids=["eventC"])
    assert np.allclose(scalars["PGA"], 0.25)
    psa = db.get_psa(event_ids=["eventC"])
    assert np.allclose(psa.to_numpy(), 0.25)
    fas = db.get_fas(event_ids=["eventC"])
    assert np.allclose(fas.to_numpy(), 0.025)
    assert db.get_site_event(event_ids=["eventC"])["rrup"].tolist() == [5.0, 6.0]
    assert db.get_realisations().loc["eventC", "rake"] == pytest.approx(90.0)
    assert db.get_events().loc["eventC", "domain_wkt"].startswith("POLYGON")


def test_merge_keeps_site_values_missing_from_incoming(db, tmp_path):
    """A NULL site value in the incoming file doesn't erase a known value here."""
    with _single_event_db(tmp_path / "rich.duckdb") as src:
        db.merge_from(src)
    assert db.get_sites().loc["siteA", "vs30"] == pytest.approx(400.0)

    # A later event that references siteA without a site characterisation
    with _single_event_db(
        tmp_path / "sparse.duckdb", event_id="eventE", site_lat=-43.8
    ) as src:
        src.con.raw_sql("UPDATE sites SET vs30 = NULL")
        db.merge_from(src)

    sites = db.get_sites()
    assert sites.loc["siteA", "vs30"] == pytest.approx(400.0)
    # Values the incoming file does have still win
    assert sites.loc["siteA", "lat"] == pytest.approx(-43.8)
    assert db.validate() == []


def test_merge_replaces_existing_event(db, tmp_path):
    """Merging an event again replaces it rather than duplicating records."""
    with _single_event_db(tmp_path / "first.duckdb", pga=0.1) as src:
        db.merge_from(src)
    with _single_event_db(tmp_path / "second.duckdb", pga=0.5) as src:
        db.merge_from(src)

    assert db.validate() == []
    assert len(db.get_records(event_ids=["eventC"])) == 4
    assert np.allclose(db.get_scalars(event_ids=["eventC"])["PGA"], 0.5)
    # Other events untouched
    assert len(db.get_records(event_ids=["eventA"])) == 12


def test_merge_without_replace_rejects_existing_event(db, tmp_path):
    """replace=False refuses to overwrite an event that already exists."""
    with _single_event_db(tmp_path / "first.duckdb") as src:
        db.merge_from(src)
    with (
        _single_event_db(tmp_path / "second.duckdb") as src,
        pytest.raises(ValueError, match="already in this database"),
    ):
        db.merge_from(src, replace=False)


def test_merge_rejects_different_period_grid(db, tmp_path):
    """pSA arrays can't be merged across different period grids."""
    src = IMDB.create(
        tmp_path / "src.duckdb", periods=[0.1, 1.0], components=COMPONENTS
    )
    with src, pytest.raises(ValueError, match="period grids differ"):
        db.merge_from(src)


def test_merge_rejects_unknown_components(db, tmp_path):
    """Components the target doesn't hold can't be merged in."""
    src = IMDB.create(
        tmp_path / "src.duckdb",
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=("rotd50",),
    )
    with src, pytest.raises(ValueError, match="components not in this database"):
        db.merge_from(src)


def test_merge_into_empty_database(tmp_path):
    """Merging into a freshly created database with the same grids works."""
    target = IMDB.create(
        tmp_path / "target.duckdb",
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=COMPONENTS,
    )
    with target, _single_event_db(tmp_path / "src.duckdb", extra_site=False) as src:
        assert target.merge_from(src) == ["eventC"]
        assert target.validate() == []
        assert list(target.get_sites().index) == ["siteA"]


def test_merge_copies_mixed_kinds_exactly(tmp_path):
    """GMM records keep their sigma, simulated ones stay sigma-free, and records without
    scalars don't gain empty scalar rows."""
    src = IMDB.create(
        tmp_path / "src.duckdb",
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=COMPONENTS,
    )
    src.add_events(pd.DataFrame({"event_id": ["eventM"]}))
    src.add_realisations(pd.DataFrame({"rel_id": ["eventM"], "event_id": ["eventM"]}))
    src.add_sites(pd.DataFrame({"site_id": ["s1"], "lat": [-43.0], "lon": [172.0]}))
    # Simulated record with scalars + pSA, and a pSA-only simulated record
    src.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventM"],
                "site_id": ["s1"],
                "component": ["000"],
                "kind": "simulated",
                "PGA": [0.1],
            }
        ),
        pSA=np.full((1, len(PERIODS)), 0.1),
    )
    src.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventM"],
                "site_id": ["s1"],
                "component": ["090"],
                "kind": "simulated",
            }
        ),
        pSA=np.full((1, len(PERIODS)), 0.2),
    )
    # GMM record with scalar and pSA sigma
    src.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventM"],
                "site_id": ["s1"],
                "component": ["000"],
                "kind": "gmm",
                "gmm_key": ["Bradley_2013"],
                "PGA": [0.08],
                "PGA_sigma": [0.6],
            }
        ),
        pSA=np.full((1, len(PERIODS)), 0.09),
        pSA_sigma=np.full((1, len(PERIODS)), 0.7),
    )

    target = IMDB.create(
        tmp_path / "target.duckdb",
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=COMPONENTS,
    )
    with src, target:
        target.merge_from(src)
        assert target.validate() == []
        for table in ("records", "scalars_ims", "psa_ims"):
            assert (
                target.con.table(table).count().to_pandas()
                == src.con.table(table).count().to_pandas()
            )
        gmm = target.get_psa(sigma=True, kind="gmm")
        assert np.allclose(gmm[f"{PERIODS[0]}_sigma"], 0.7)
        assert target.get_scalars(sigma=True, kind="gmm")["PGA_sigma"].iloc[0] == (
            pytest.approx(0.6)
        )
