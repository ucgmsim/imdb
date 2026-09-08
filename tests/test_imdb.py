import numpy as np
import pandas as pd
import pytest

from imdb import IMDB, schema

from .conftest import (
    COMPONENTS,
    EVENTS,
    FREQUENCIES,
    PERIODS,
    SCALARS,
    SITES,
    build_frames,
)

N_RECORDS = 4 * len(SITES) * len(COMPONENTS)


def test_create_populates_documentation(db):
    assert db.db_meta["schema_version"] == schema.SCHEMA_VERSION
    assert db.db_meta["dataset_description"] == "test fixture"
    assert db.db_meta["n_periods"] == str(len(PERIODS))
    assert db.components == COMPONENTS
    assert db.notes == schema.NOTES
    assert db.im_units == schema.IM_UNITS
    assert db.periods.tolist() == PERIODS
    assert db.frequencies.tolist() == FREQUENCIES
    assert db.periods.index.tolist() == [1, 2, 3, 4, 5]


def test_dimension_reads(db):
    assert sorted(db.get_events()["event_id"]) == EVENTS
    assert len(db.get_realisations()) == 4
    assert sorted(db.get_sites()["site_id"]) == SITES
    assert len(db.get_site_event()) == len(SITES) * len(EVENTS)
    assert len(db.get_records()) == N_RECORDS


def test_metadata_expansion(db):
    events = db.get_events(expand_metadata=True)
    assert "metadata" not in events.columns
    assert set(events["fault_type"]) == {"DS_POINT_SOURCE", "NORMAL_FAULTING"}
    assert db.db_meta["event_metadata_keys"] == "fault_type"
    assert db.db_meta["site_metadata_keys"] == "elevation"


def test_psa_and_fas_round_trip(db):
    _, _, _, _, records = build_frames()
    order = db.get_records().sort_index()

    psa = db.get_psa()
    assert psa.columns.tolist() == PERIODS
    fas = db.get_fas()
    assert fas.columns.tolist() == FREQUENCIES

    # rebuild the input keyed the same way the database is
    key = ["rel_id", "site_id", "component"]
    expected = records.set_index(key)
    got = order.reset_index().set_index(key)

    for rid, k in zip(got["record_id"], got.index, strict=True):
        np.testing.assert_array_equal(
            psa.loc[rid].to_numpy(dtype=np.float32), expected.loc[k, "pSA"]
        )
        np.testing.assert_array_equal(
            fas.loc[rid].to_numpy(dtype=np.float32), expected.loc[k, "FAS"]
        )


def test_scalar_round_trip_and_rotd_nulls(db):
    _, _, _, _, records = build_frames()
    scalars = db.get_scalars()
    assert scalars.columns.tolist() == list(schema.SCALAR_IMS)

    recs = db.get_records()
    joined = recs.join(scalars)
    expected = records.set_index(["rel_id", "site_id", "component"])

    for record_id, row in joined.iterrows():
        key = (row["rel_id"], row["site_id"], row["component"])
        for im in SCALARS:
            if im in schema.ROTD_UNDEFINED and row["component"].startswith("rotd"):
                assert pd.isna(row[im]), (record_id, im)
            else:
                assert row[im] == pytest.approx(expected.loc[key, im], rel=1e-6)


def test_subset_of_periods(db):
    full = db.get_psa()
    subset = db.get_psa(periods=[1.0, 0.01])
    assert subset.columns.tolist() == [1.0, 0.01]
    pd.testing.assert_series_equal(subset[1.0], full[1.0], check_names=False)


def test_period_written_with_p(db):
    df = db.get_im_df(["pSA_0p1", "pSA_10p0"])
    assert df.columns.tolist() == ["pSA_0p1", "pSA_10p0"]
    np.testing.assert_allclose(df["pSA_0p1"], db.get_psa(periods=[0.1])[0.1])


def test_get_im_df_matches_typed_calls(db):
    names = ["PGA", "pSA_1.0", "FAS_10.0", "Ds595"]
    df = db.get_im_df(names)
    assert df.columns.tolist() == names
    assert len(df) == N_RECORDS
    np.testing.assert_allclose(df["PGA"], db.get_scalars(ims=["PGA"])["PGA"])
    np.testing.assert_allclose(df["pSA_1.0"], db.get_psa(periods=[1.0])[1.0])
    np.testing.assert_allclose(df["FAS_10.0"], db.get_fas(frequencies=[10.0])[10.0])


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, N_RECORDS),
        ({"events": ["ev1"]}, 2 * len(SITES) * len(COMPONENTS)),
        ({"sites": ["stnA"]}, 4 * len(COMPONENTS)),
        ({"rels": ["ev1_REL01"]}, len(SITES) * len(COMPONENTS)),
        ({"component": "rotd50"}, N_RECORDS // 2),
        ({"component": ["geom", "rotd50"]}, N_RECORDS),
        ({"events": ["ev1"], "component": "geom"}, 2 * len(SITES)),
        ({"max_rrup": 30.0}, 3 * 2 * len(COMPONENTS)),
    ],
)
def test_filters(db, filters, expected):
    assert len(db.get_records(**filters)) == expected
    assert len(db.get_psa(periods=[1.0], **filters)) == expected
    assert len(db.get_scalars(ims=["PGA"], **filters)) == expected
    assert len(db.get_im_df(["PGA", "pSA_1.0"], **filters)) == expected


def test_record_ids_filter(db):
    wanted = db.get_records().index[:5].to_numpy()
    got = db.get_psa(periods=[1.0], record_ids=wanted)
    assert sorted(got.index) == sorted(wanted)


def test_filters_match_raw_sql(db):
    got = db.get_records(events=["ev2"], component="geom").index.tolist()
    expected = [
        row[0]
        for row in db.sql(
            """
            SELECT r.record_id FROM records r
            JOIN events e ON e.event_int_id = r.event_int_id
            WHERE e.event_id = 'ev2' AND r.component = 'geom'
            """
        ).fetchall()
    ]
    assert sorted(got) == sorted(expected)


def test_site_event_filters(db):
    assert len(db.get_site_event(sites=["stnA"])) == 2
    assert len(db.get_site_event(events=["ev1"])) == 3
    assert len(db.get_site_event(max_rrup=30.0)) == 3
    expanded = db.get_site_event(sites=["stnA"], expand_metadata=True)
    assert "metadata" not in expanded.columns


def test_bad_requests_raise(db):
    with pytest.raises(KeyError, match="not on this database's grid"):
        db.get_psa(periods=[2.5])
    with pytest.raises(ValueError, match="cannot parse IM name"):
        db.get_im_df(["SA_1.0"])
    with pytest.raises(ValueError, match="not in this database"):
        db.get_records(component="000")
    with pytest.raises(ValueError, match="unknown scalar IMs"):
        db.get_scalars(ims=["MMI"])
    with pytest.raises(KeyError, match="unknown event_id"):
        db.get_records(events=["nope"])


def test_read_only_rejects_writes(db):
    with pytest.raises(PermissionError):
        db.set_db_meta({"source": "nope"})


def test_validate_is_clean(db):
    assert db.validate() == []


def test_validate_finds_orphan_record(wdb):
    wdb.conn.execute("INSERT INTO records VALUES (9999, 1, 999, 1, 'geom')")
    problems = wdb.validate()
    assert any("orphan rel_int_id" in p for p in problems)


def test_validate_finds_wrong_array_length(wdb):
    rid = int(wdb.get_records().index[0])
    wdb.conn.execute("UPDATE psa_ims SET pSA = [1.0, 2.0] WHERE record_id = ?", [rid])
    assert any("pSA length is wrong" in p for p in wdb.validate())


def test_validate_finds_undeclared_metadata_key(wdb):
    wdb.conn.execute(
        """UPDATE sites SET metadata = '{"basin": "Canterbury"}' WHERE site_int_id = 1"""
    )
    assert any(
        "not declared in db_meta.site_metadata_keys" in p for p in wdb.validate()
    )


def test_validate_finds_duplicate_record_key(wdb):
    wdb.conn.execute(
        "INSERT INTO records SELECT 9999, event_int_id, rel_int_id, site_int_id, component "
        "FROM records LIMIT 1"
    )
    assert any("duplicated (rel_int_id" in p for p in wdb.validate())


def test_undeclared_metadata_key_rejected_on_write(wdb):
    sites = pd.DataFrame(
        {
            "site_id": ["stnD"],
            "lat": [-42.0],
            "lon": [173.0],
            "metadata": [{"basin": "Canterbury"}],
        }
    )
    with pytest.raises(ValueError, match="not declared in db_meta.site_metadata_keys"):
        wdb.add_sites(sites)


def test_unknown_column_rejected_on_write(wdb):
    with pytest.raises(ValueError, match="not in sites"):
        wdb.add_sites(
            pd.DataFrame(
                {"site_id": ["stnD"], "lat": [-42.0], "lon": [173.0], "vs20": [1.0]}
            )
        )


def test_duplicate_id_rejected_on_write(wdb):
    with pytest.raises(ValueError, match="already in the database"):
        wdb.add_sites(
            pd.DataFrame({"site_id": ["stnA"], "lat": [-42.0], "lon": [173.0]})
        )


def test_undeclared_component_rejected_on_write(wdb):
    _, _, _, _, records = build_frames()
    bad = records.head(1).copy()
    bad["component"] = "000"
    with pytest.raises(ValueError, match="not declared in db_meta.components"):
        wdb.add_records(bad)


def test_wrong_array_length_rejected_on_write(wdb):
    _, _, _, _, records = build_frames()
    bad = records.head(1).copy()
    bad["rel_id"] = "ev1_REL01"
    bad["pSA"] = [np.ones(3, dtype=np.float32)]
    with pytest.raises(ValueError, match="grid has 5 entries"):
        wdb.add_records(bad)


def test_delete_event_then_readd(db_path):
    events, rels, _, site_event, records = build_frames()
    with IMDB(db_path, read_only=False) as db:
        before = db.get_im_df(["PGA", "pSA_1.0"], events=["ev1"]).to_numpy()
        old_ids = set(db.get_records(events=["ev1"]).index)

        db.delete_event("ev1")
        assert db.validate() == []
        assert len(db.get_records()) == N_RECORDS // 2
        assert len(db.get_site_event()) == len(SITES)

        keep = rels["event_id"] == "ev1"
        db.add_events(events[events["event_id"] == "ev1"])
        db.add_realisations(rels[keep])
        db.add_site_event(site_event[site_event["event_id"] == "ev1"])
        db.add_records(records[records["rel_id"].isin(rels.loc[keep, "rel_id"])])
        db.finalise()

        after = db.get_im_df(["PGA", "pSA_1.0"], events=["ev1"]).to_numpy()
        new_ids = set(db.get_records(events=["ev1"]).index)

    np.testing.assert_allclose(np.sort(before, axis=0), np.sort(after, axis=0))
    assert not (old_ids & new_ids)


def test_records_without_all_im_tables(tmp_path):
    path = tmp_path / "psa_only.duckdb"
    events, rels, sites, site_event, records = build_frames()
    with IMDB.create(path, periods=PERIODS, components=COMPONENTS) as db:
        db.add_events(events)
        db.add_realisations(rels)
        db.add_sites(sites)
        db.add_site_event(site_event)
        db.add_records(records[["rel_id", "site_id", "component", "pSA"]])
        db.finalise()

    with IMDB(path) as db:
        assert len(db.get_psa()) == N_RECORDS
        assert db.get_scalars().empty
        assert len(db.frequencies) == 0
        with pytest.raises(KeyError, match="no FAS grid"):
            db.get_fas()


def test_create_refuses_to_clobber(db_path):
    with pytest.raises(FileExistsError):
        IMDB.create(db_path, periods=PERIODS)
    IMDB.create(db_path, periods=PERIODS, overwrite=True).close()


def test_finalise_raises_on_inconsistency(wdb):
    wdb.conn.execute("INSERT INTO records VALUES (9999, 1, 999, 1, 'geom')")
    with pytest.raises(ValueError, match="database is inconsistent"):
        wdb.finalise()
