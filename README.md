# imdb

A library for reading and writing intensity measure databases (IMDBs), DuckDB
databases of simulated ground-motion intensity measures. Schema is documented in
`imdb/schema.py`.

## Usage

```python
from imdb import IMDB

# read
with IMDB("run_set.duckdb") as db:
    records = db.get_records(event_ids=["event1"], component="rotd50")
    psa = db.get_psa(periods=[0.1, 1.0], event_ids=["event1"])
    scalars = db.get_scalars(ims=["PGA", "PGV"])

# write
db = IMDB.create("new.duckdb", periods=[0.1, 0.2, 1.0])
db.add_events(events_df)
db.add_realisations(realisations_df)
db.add_sites(sites_df)
db.add_site_event(site_event_df)
db.add_records(records_df)  # rel_id, site_id, component, pSA, FAS, scalar IM columns
db.validate()
db.close()
```

## Development

```
uv sync --all-groups
uv run pytest -q
uv run ruff check
uv run ruff format
uv run ty check
```
