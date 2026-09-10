"""Basic tests: the library works for its intended, correct usage."""

import ibis
import numpy as np
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


def test_add_records_rejects_duplicate_within_df(db):
    with pytest.raises(ValueError, match="duplicate"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0", "eventA_rel0"],
                    "site_id": ["siteA", "siteA"],
                    "component": ["000", "000"],
                    "kind": ["observed", "observed"],
                }
            )
        )


def test_add_records_rejects_duplicate_of_existing_record(db):
    with pytest.raises(ValueError, match="already exist"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["simulated"],
                }
            )
        )


def test_validate_catches_sigma_on_non_gmm_record(db):
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0"],
                "site_id": ["siteA"],
                "component": ["000"],
                "kind": ["observed"],
                "PGA": [0.5],
                "PGA_sigma": [0.6],
            }
        )
    )

    problems = db.validate()
    assert any("scalars_ims.PGA_sigma" in p for p in problems)


def test_all_nan_row_not_written_to_psa(db):
    record_int_ids = db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0", "eventA_rel0"],
                "site_id": ["siteA", "siteB"],
                "component": ["000", "000"],
                "kind": ["observed", "observed"],
            }
        ),
        pSA=np.array([[1.0] * len(PERIODS), [np.nan] * len(PERIODS)]),
    )

    psa = db.get_psa(record_int_ids=list(record_int_ids))
    assert list(psa.index) == [record_int_ids[0]]


def test_get_site_event_filters(db):
    by_event = db.get_site_event(event_ids=["eventA"])
    assert (by_event["event_id"] == "eventA").all()
    assert len(by_event) == len(SITES)

    by_rrup = db.get_site_event(max_rrup=1.5)
    assert (by_rrup["rrup"] <= 1.5).all()
    assert len(by_rrup) == 2


def test_rotd_component_nulls_undefined_scalars(tmp_path):
    db = IMDB.create(tmp_path / "rotd.duckdb", periods=[], components=("rotd50",))
    db.add_events(pd.DataFrame({"event_id": ["e1"]}))
    db.add_realisations(pd.DataFrame({"rel_id": ["r1"], "event_id": ["e1"]}))
    db.add_sites(pd.DataFrame({"site_id": ["s1"], "lat": [-43.5], "lon": [172.6]}))

    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["r1"],
                "site_id": ["s1"],
                "component": ["rotd50"],
                "kind": ["simulated"],
                "PGA": [0.5],
                "CAV": [1.2],
            }
        )
    )

    scalars = db.get_scalars(ims=["PGA", "CAV"])
    assert scalars["PGA"].iloc[0] == 0.5
    assert pd.isna(scalars["CAV"].iloc[0])
    db.close()
