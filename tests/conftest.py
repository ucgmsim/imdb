"""Shared fixtures: one small synthetic IMDB."""

import numpy as np
import pandas as pd
import pytest

from imdb import IMDB

PERIODS = [0.1, 0.2, 0.5, 1.0, 2.0]
FREQUENCIES = [1.0, 5.0, 10.0]
COMPONENTS = ["000", "090"]
EVENTS = ["eventA", "eventB"]
SITES = ["siteA", "siteB", "siteC"]


@pytest.fixture
def db(tmp_path):
    """A small, fully populated IMDB: 2 events, 2 realisations each, 3 sites, 2 components."""
    path = tmp_path / "test.duckdb"
    db = IMDB.create(
        path, periods=PERIODS, frequencies=FREQUENCIES, components=COMPONENTS
    )

    db.add_events(pd.DataFrame({"event_id": EVENTS, "magnitude": [6.0, 7.0]}))

    rel_ids = [f"{e}_rel{i}" for e in EVENTS for i in range(2)]
    db.add_realisations(
        pd.DataFrame(
            {
                "rel_id": rel_ids,
                "event_id": [e for e in EVENTS for _ in range(2)],
            }
        )
    )

    db.add_sites(
        pd.DataFrame(
            {
                "site_id": SITES,
                "lat": [-43.5, -43.6, -43.7],
                "lon": [172.6, 172.7, 172.8],
            }
        )
    )

    db.add_site_event(
        pd.DataFrame(
            {
                "site_id": [s for _ in EVENTS for s in SITES],
                "event_id": [e for e in EVENTS for _ in SITES],
                "rrup": np.arange(len(EVENTS) * len(SITES), dtype=float),
            }
        )
    )

    rng = np.random.default_rng(0)
    rows = [
        (rel_id, site_id, component)
        for rel_id in rel_ids
        for site_id in SITES
        for component in COMPONENTS
    ]
    n = len(rows)
    db.add_records(
        pd.DataFrame(
            {
                "rel_id": [r[0] for r in rows],
                "site_id": [r[1] for r in rows],
                "component": [r[2] for r in rows],
                "pSA": [rng.uniform(size=len(PERIODS)) for _ in range(n)],
                "FAS": [rng.uniform(size=len(FREQUENCIES)) for _ in range(n)],
                "PGA": rng.uniform(size=n),
                "PGV": rng.uniform(size=n),
                "PGD": rng.uniform(size=n),
            }
        )
    )
    yield db
    db.close()
