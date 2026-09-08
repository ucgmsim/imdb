import numpy as np
import pandas as pd
import pytest

from imdb import IMDB

PERIODS = [0.01, 0.1, 1.0, 3.0, 10.0]
FREQUENCIES = [0.1, 1.0, 10.0]
COMPONENTS = ["geom", "rotd50"]
EVENTS = ["ev1", "ev2"]
SITES = ["stnA", "stnB", "stnC"]
SCALARS = ["PGA", "PGV", "PGD", "CAV", "AI", "Ds575", "Ds595"]


def build_frames():
    """Deterministic input frames covering every table."""
    events = pd.DataFrame(
        {
            "event_id": EVENTS,
            "magnitude": [7.1, 6.2],
            "tect_type": ["SUBDUCTION_SLAB", "ACTIVE_SHALLOW"],
            "dtop": [30.0, 0.5],
            "metadata": [
                {"fault_type": "DS_POINT_SOURCE"},
                {"fault_type": "NORMAL_FAULTING"},
            ],
        }
    )
    rels = pd.DataFrame(
        {
            "rel_id": [f"{e}_REL{i:02d}" for e in EVENTS for i in (1, 2)],
            "event_id": [e for e in EVENTS for _ in (1, 2)],
            "magnitude": [7.1, 7.12, 6.2, 6.18],
            "rake": [90.0, 88.0, -90.0, -92.0],
            "hypo_lat": [-43.5, -43.6, -41.2, -41.3],
            "hypo_lon": [172.6, 172.7, 174.8, 174.9],
            "hypo_depth": [40.0, 42.0, 8.0, 9.0],
            "metadata": [{"solver": "emod3d"}] * 4,
        }
    )
    sites = pd.DataFrame(
        {
            "site_id": SITES,
            "lat": [-43.5, -43.6, -41.3],
            "lon": [172.6, 172.7, 174.8],
            "vs30": [300.0, 500.0, 250.0],
            "z1p0": [0.3, 0.1, 0.5],
            "metadata": [{"elevation": 10.0}, {"elevation": 55.0}, {"elevation": 3.0}],
        }
    )
    site_event = pd.DataFrame(
        {
            "site_id": [s for s in SITES for _ in EVENTS],
            "event_id": EVENTS * len(SITES),
            "rrup": [10.0, 300.0, 25.0, 280.0, 400.0, 5.0],
            "rjb": [8.0, 295.0, 22.0, 275.0, 395.0, 3.0],
        }
    )

    rows = []
    for rel_id in rels["rel_id"]:
        for site_id in SITES:
            for component in COMPONENTS:
                base = float(len(rows) + 1)
                row = {
                    "rel_id": rel_id,
                    "site_id": site_id,
                    "component": component,
                    "pSA": np.array(
                        [base + i / 8 for i in range(1, len(PERIODS) + 1)],
                        dtype=np.float32,
                    ),
                    "FAS": np.array(
                        [base * 2 + i / 8 for i in range(1, len(FREQUENCIES) + 1)],
                        dtype=np.float32,
                    ),
                }
                for k, im in enumerate(SCALARS):
                    row[im] = base + (k + 1) / 16
                rows.append(row)
    records = pd.DataFrame(rows)
    return events, rels, sites, site_event, records


@pytest.fixture
def db_path(tmp_path):
    """Path of a small, fully populated database."""
    path = tmp_path / "test_ims.duckdb"
    events, rels, sites, site_event, records = build_frames()
    with IMDB.create(
        path,
        periods=PERIODS,
        frequencies=FREQUENCIES,
        components=COMPONENTS,
        db_meta={"dataset_description": "test fixture", "source": "conftest"},
    ) as db:
        db.add_events(events)
        db.add_realisations(rels)
        db.add_sites(sites)
        db.add_site_event(site_event)
        db.add_records(records)
        db.finalise()
    return path


@pytest.fixture
def db(db_path):
    """The fixture database, open read-only."""
    with IMDB(db_path) as handle:
        yield handle


@pytest.fixture
def wdb(db_path):
    """The fixture database, open for writing."""
    with IMDB(db_path, read_only=False) as handle:
        yield handle
