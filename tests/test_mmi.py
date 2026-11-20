"""Tests for the MMI scalar IM (schema version 3) and reading older databases without it."""

import ibis
import pandas as pd
import pytest

from imdb import IMDB, schema
from tests.conftest import COMPONENTS, FREQUENCIES, PERIODS


def _one_site_db(path, components=("rotd50", "geom")):
    db = IMDB.create(path, periods=[], components=components)
    db.add_events(pd.DataFrame({"event_id": ["e1"]}))
    db.add_realisations(pd.DataFrame({"rel_id": ["r1"], "event_id": ["e1"]}))
    db.add_sites(pd.DataFrame({"site_id": ["s1"], "lat": [-43.5], "lon": [172.6]}))
    return db


def _downgrade_to_v2(path):
    """Make a database look like one built under schema version 2: no MMI columns."""
    con = ibis.duckdb.connect(path, read_only=False)
    con.raw_sql("ALTER TABLE scalars_ims DROP COLUMN MMI_sigma")
    con.raw_sql("ALTER TABLE scalars_ims DROP COLUMN MMI")
    con.raw_sql("DELETE FROM im_units WHERE im = 'MMI'")
    con.raw_sql("UPDATE db_meta SET value = '2' WHERE key = 'schema_version'")
    con.disconnect()


def test_mmi_is_a_scalar_im_with_a_unit(tmp_path):
    assert "MMI" in schema.SCALAR_IMS
    assert "MMI" not in schema.ROTD_UNDEFINED
    with IMDB.create(tmp_path / "units.duckdb", periods=[]) as db:
        units = db.con.table("im_units").to_pandas().set_index("im")["unit"]
        assert units["MMI"] == "1"


def test_mmi_round_trips_on_every_component_including_rotd(tmp_path):
    with _one_site_db(tmp_path / "mmi.duckdb") as db:
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["r1", "r1"],
                    "site_id": ["s1", "s1"],
                    "component": ["rotd50", "geom"],
                    "kind": "simulated",
                    "PGV": [10.0, 9.0],
                    "MMI": [5.5, 5.25],
                    "CAV": [1.0, 1.0],
                }
            )
        )
        scalars = db.get_scalars().sort_values("MMI")
        assert scalars["MMI"].tolist() == [5.25, 5.5]
        # MMI, unlike CAV, is defined on rotd components
        assert scalars["CAV"].isna().sum() == 1
        assert db.validate() == []


def test_gmm_mmi_sigma(tmp_path):
    with _one_site_db(tmp_path / "gmm.duckdb") as db:
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["r1"],
                    "site_id": ["s1"],
                    "component": ["rotd50"],
                    "kind": ["gmm"],
                    "gmm_key": ["TestGMICE"],
                    "MMI": [6.0],
                    "MMI_sigma": [0.7],
                }
            )
        )
        scalars = db.get_scalars(ims=["MMI"], sigma=True)
        assert scalars["MMI"].iloc[0] == 6.0
        assert scalars["MMI_sigma"].iloc[0] == pytest.approx(0.7)
        assert db.validate() == []


def test_validate_catches_mmi_sigma_on_simulated_record(tmp_path):
    with _one_site_db(tmp_path / "bad.duckdb") as db:
        db.add_records(
            pd.DataFrame(
                {
                    "rel_id": ["r1"],
                    "site_id": ["s1"],
                    "component": ["geom"],
                    "kind": ["simulated"],
                    "MMI": [6.0],
                    "MMI_sigma": [0.7],
                }
            )
        )
        assert any("scalars_ims.MMI_sigma" in p for p in db.validate())


def test_schema_v2_is_read_compatible():
    assert schema.SCHEMA_VERSION == "3"
    assert {"1", "2"} <= schema.READ_COMPATIBLE_SCHEMA_VERSIONS


def test_v2_database_reads_without_mmi(db):
    path = db.path
    db.close()
    _downgrade_to_v2(path)

    with IMDB(path) as reader:
        assert reader.db_meta["schema_version"] == "2"
        scalars = reader.get_scalars(sigma=True)
        assert "MMI" not in scalars.columns
        assert "MMI_sigma" not in scalars.columns
        assert {"PGA", "PGV", "PGD", "PGA_sigma"} <= set(scalars.columns)
        assert reader.validate() == []

    with pytest.raises(RuntimeError, match="schema version"):
        IMDB(path, read_only=False).open()


def test_merge_v2_database_into_v3_leaves_mmi_null(db, tmp_path):
    source = db.path
    expected = db.get_scalars(ims=["PGA"])
    db.close()
    _downgrade_to_v2(source)

    target = IMDB.create(
        tmp_path / "target.duckdb",
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=COMPONENTS,
    )
    with target, IMDB(source) as other:
        target.merge_from(other)
        scalars = target.get_scalars()
        assert scalars["MMI"].isna().all()
        assert sorted(scalars["PGA"]) == sorted(expected["PGA"])
        assert target.validate() == []
