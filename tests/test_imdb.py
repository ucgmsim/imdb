"""Basic tests: the library works for its intended, correct usage."""

import ibis
import numpy as np
import pandas as pd
import pytest

from imdb import IMDB
from tests.conftest import COMPONENTS, EVENTS, FREQUENCIES, PERIODS, SITES


def test_create_rejects_existing_path(tmp_path):
    path = tmp_path / "existing.duckdb"
    IMDB.create(path, periods=[]).close()

    with pytest.raises(FileExistsError):
        IMDB.create(path, periods=[])


def test_create_db_meta_cannot_override_reserved_keys(tmp_path):
    path = tmp_path / "reserved.duckdb"
    db = IMDB.create(path, periods=[], db_meta={"schema_version": "stale"})
    assert db.db_meta["schema_version"] == "1"
    db.close()


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


def test_add_site_event_rejects_duplicate_within_df(db):
    with pytest.raises(ValueError, match="duplicate"):
        db.add_site_event(
            pd.DataFrame(
                {
                    "site_id": ["siteA", "siteA"],
                    "event_id": ["eventA", "eventA"],
                    "rrup": [1.0, 1.0],
                }
            )
        )


def test_add_site_event_rejects_duplicate_of_existing_row(db):
    with pytest.raises(ValueError, match="already exist"):
        db.add_site_event(
            pd.DataFrame({"site_id": ["siteA"], "event_id": ["eventA"], "rrup": [1.0]})
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


def test_con_raises_when_not_open(tmp_path):
    db = IMDB(tmp_path / "unopened.duckdb")
    with pytest.raises(RuntimeError, match="not open"):
        _ = db.con


def test_context_manager(tmp_path):
    path = tmp_path / "ctx.duckdb"
    with IMDB.create(path, periods=[]) as db:
        assert db.con is not None
    assert db._con is None


def test_add_records_requires_kind_column(db):
    with pytest.raises(ValueError, match="kind"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                }
            )
        )


def test_add_records_rejects_unknown_component(db):
    with pytest.raises(ValueError, match="components not in this database"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["ver"],
                    "kind": ["observed"],
                }
            )
        )


def test_add_records_rejects_bad_psa_shape(db):
    with pytest.raises(ValueError, match="pSA must have shape"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            pSA=np.zeros((2, len(PERIODS))),
        )


def test_add_records_rejects_bad_fas_shape(db):
    with pytest.raises(ValueError, match="FAS must have shape"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            FAS=np.zeros((2, len(FREQUENCIES))),
        )


def test_add_records_rejects_wrong_psa_grid_length(db):
    with pytest.raises(ValueError, match="pSA must have shape"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            pSA=np.zeros((1, len(PERIODS) - 1)),
        )


def test_add_records_rejects_wrong_fas_grid_length(db):
    with pytest.raises(ValueError, match="FAS must have shape"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            FAS=np.zeros((1, len(FREQUENCIES) - 1)),
        )


def test_add_records_rejects_psa_sigma_without_psa(db):
    with pytest.raises(ValueError, match="pSA_sigma must match pSA"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            pSA_sigma=np.zeros((1, len(PERIODS))),
        )


def test_add_records_rejects_fas_sigma_without_fas(db):
    with pytest.raises(ValueError, match="FAS_sigma must match FAS"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                }
            ),
            FAS_sigma=np.zeros((1, len(FREQUENCIES))),
        )


def test_add_records_masks_rotd_sigma(tmp_path):
    db = IMDB.create(tmp_path / "rotd_sigma.duckdb", periods=[], components=("rotd50",))
    db.add_events(pd.DataFrame({"event_id": ["e1"]}))
    db.add_realisations(pd.DataFrame({"rel_id": ["r1"], "event_id": ["e1"]}))
    db.add_sites(pd.DataFrame({"site_id": ["s1"], "lat": [-43.5], "lon": [172.6]}))

    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["r1"],
                "site_id": ["s1"],
                "component": ["rotd50"],
                "kind": ["gmm"],
                "gmm_key": ["TestGMM2020"],
                "PGA": [0.5],
                "CAV": [1.2],
                "CAV_sigma": [0.3],
            }
        )
    )

    scalars = db.get_scalars(ims=["PGA", "CAV"], sigma=True)
    assert scalars["PGA"].iloc[0] == 0.5
    assert pd.isna(scalars["CAV"].iloc[0])
    assert pd.isna(scalars["CAV_sigma"].iloc[0])
    db.close()


def test_add_records_rolls_back_on_failure(db, monkeypatch):
    def broken_insert(table, data):
        if table == "scalars_ims":
            raise RuntimeError("boom")
        return real_insert(table, data)

    real_insert = db.con.insert
    monkeypatch.setattr(db.con, "insert", broken_insert)

    with pytest.raises(RuntimeError, match="boom"):
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["eventA_rel0"],
                    "site_id": ["siteA"],
                    "component": ["000"],
                    "kind": ["observed"],
                    "PGA": [0.5],
                }
            )
        )

    monkeypatch.undo()
    assert len(db.get_records(rel_ids=["eventA_rel0"], site_ids=["siteA"])) == 2


def test_validate_detects_duplicate_record_int_id(db):
    some_id = db.get_records().index[0]
    db.con.raw_sql(f"INSERT INTO psa_ims (record_int_id, pSA) VALUES ({some_id}, NULL)")
    problems = db.validate()
    assert any("record_int_id is not unique" in p for p in problems)


def test_validate_detects_unknown_foreign_key(db):
    db.con.raw_sql(
        "INSERT INTO records (record_int_id, event_int_id, rel_int_id, site_int_id, "
        "component, kind) VALUES (999999, 999999, 999999, 999999, '000', 'observed')"
    )
    problems = db.validate()
    assert any("unknown event_int_id" in p for p in problems)


def test_validate_detects_orphan_im_row(db):
    db.con.raw_sql("INSERT INTO psa_ims (record_int_id, pSA) VALUES (999999, NULL)")
    problems = db.validate()
    assert any("psa_ims: 1 rows with an unknown record_int_id" in p for p in problems)


def test_validate_detects_bad_gmm_key(db):
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0"],
                "site_id": ["siteA"],
                "component": ["000"],
                "kind": ["observed"],
                "gmm_key": ["shouldnt be set"],
            }
        )
    )
    problems = db.validate()
    assert any("gmm_key inconsistent with kind" in p for p in problems)


def test_get_sites(db):
    sites = db.get_sites()
    assert set(sites.index) == set(SITES)


def test_get_site_event_filters_by_site_ids(db):
    by_site = db.get_site_event(site_ids=["siteA"])
    assert (by_site["site_id"] == "siteA").all()
    assert len(by_site) == len(EVENTS)


def test_get_records_filters(db):
    by_rel = db.get_records(rel_ids=["eventA_rel0"])
    assert (by_rel["rel_id"] == "eventA_rel0").all()

    by_site = db.get_records(site_ids=["siteA"])
    assert (by_site["site_id"] == "siteA").all()

    by_component = db.get_records(component="000")
    assert (by_component["component"] == "000").all()

    db.add_records(
        pd.DataFrame(
            {
                "rel_id": ["eventA_rel0"],
                "site_id": ["siteA"],
                "component": ["090"],
                "kind": ["gmm"],
                "gmm_key": ["TestGMM2020"],
            }
        )
    )
    by_gmm_key = db.get_records(gmm_key="TestGMM2020")
    assert (by_gmm_key["gmm_key"] == "TestGMM2020").all()


def test_get_psa_and_fas_sigma(db):
    psa = db.get_psa(sigma=True)
    assert all(f"{p}_sigma" in psa.columns for p in PERIODS)

    fas = db.get_fas(sigma=True)
    assert all(f"{f}_sigma" in fas.columns for f in FREQUENCIES)


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
