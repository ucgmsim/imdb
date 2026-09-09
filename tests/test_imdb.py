"""Basic tests: the library works for its intended, correct usage."""

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
