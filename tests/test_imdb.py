"""Basic tests: the library works for its intended, correct usage."""

import ibis
import pandas as pd
import pytest

from imdb import IMDB
from tests.conftest import COMPONENTS, EVENTS, FREQUENCIES, PERIODS, SITES


def test_round_trip(db):
    records = db.get_records()
    n = len(EVENTS) * 2 * len(SITES) * len(COMPONENTS)
    assert len(records) == n

    psa = db.get_psa()
    assert list(psa.columns) == [str(p) for p in PERIODS]
    assert (psa.index == records.index).all()

    fas = db.get_fas()
    assert list(fas.columns) == [str(f) for f in FREQUENCIES]

    scalars = db.get_scalars(ims=["PGA", "PGV", "PGD"])
    assert set(scalars.columns) == {"PGA", "PGV", "PGD"}
    assert not scalars.isna().any().any()


def test_filter_by_event(db):
    records = db.get_records(event_ids=["eventA"])
    assert (records["event_id"] == "eventA").all()
    assert len(records) == 2 * len(SITES) * len(COMPONENTS)


def test_validate_clean(db):
    assert db.validate() == []


def test_delete_and_readd(db):
    db.delete_event("eventA")
    assert len(db.get_records(event_ids=["eventA"])) == 0
    assert len(db.get_realisations()) == 2
    assert db.validate() == []
    assert "eventA" not in db.get_events().index

    db.add_events(pd.DataFrame({"event_id": ["eventA"], "magnitude": [6.0]}))
    assert "eventA" in db.get_events().index


def test_gmm_records(db):
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0"],
                "site_id": ["siteA"],
                "component": ["000"],
                "kind": ["gmm"],
                "gmm_key": ["TestGMM2020"],
                "PGA": [0.5],
                "PGA_sigma": [0.6],
            }
        )
    )

    assert db.validate() == []

    gmm_records = db.get_records(kind="gmm")
    assert len(gmm_records) == 1
    assert gmm_records["gmm_key"].iloc[0] == "TestGMM2020"

    scalars = db.get_scalars(ims=["PGA"], sigma=True, kind="gmm")
    assert scalars["PGA"].iloc[0] == 0.5
    assert scalars["PGA_sigma"].iloc[0] == 0.6

    simulated_records = db.get_records(kind="simulated")
    assert (simulated_records["gmm_key"].isna()).all()


def test_schema_version_mismatch_rejected(db):
    path = db.path
    db.close()

    con = ibis.duckdb.connect(path, read_only=False)
    con.raw_sql("UPDATE db_meta SET value = 'stale' WHERE key = 'schema_version'")
    con.disconnect()

    with pytest.raises(RuntimeError, match="schema version"):
        IMDB(path, read_only=True).open()


def test_validate_catches_sigma_on_non_gmm_record(db):
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0"],
                "site_id": ["siteA"],
                "component": ["000"],
                "kind": ["simulated"],
                "PGA": [0.5],
                "PGA_sigma": [0.6],
            }
        )
    )

    problems = db.validate()
    assert any("scalars_ims.PGA_sigma" in p for p in problems)
