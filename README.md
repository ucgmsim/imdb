# imdb

Reading and writing intensity measure databases (IMDBs): DuckDB files holding
intensity measures from physics-based ground-motion simulation.

One file holds one simulation run set. Every file uses the same schema and is
self-contained; combine several with `ATTACH` and `UNION ALL`.

## Schema

Thirteen tables: four dimensions (`events`, `realisations`, `sites`,
`site_event`), one identity table (`records`), three IM tables (`psa_ims`,
`fas_ims`, `scalars_ims`), two vocabulary tables (`periods`, `frequencies`) and
three documentation tables (`db_meta`, `im_units`, `notes`).

A ground motion is identified by `(rel_id, site_id, component)`. pSA and FAS are
stored as one `FLOAT[]` per record, indexed by `periods.period_index` and
`frequencies.freq_index` (both 1-based). Scalar IMs are named columns.

`imdb/schema.py` holds the DDL and is the single source of truth. Every database
also documents itself: read the `notes` and `db_meta` tables.

## Reading

```python
from imdb import IMDB

with IMDB("cs200.duckdb") as db:
    df = db.get_im_df(
        ["PGA", "pSA_0.1", "pSA_1.0", "FAS_5.0"],
        events=["AlpineF2K"], component="rotd50", max_rrup=200,
    )
```

`get_im_df` returns one DataFrame indexed by `record_id`, columns named as
requested. Underneath it are `get_psa(periods=...)`, `get_fas(frequencies=...)`
and `get_scalars(ims=...)`, which label their columns with the period in
seconds, the frequency in Hz, and the IM name respectively.

Every read takes the same keyword filters: `events`, `rels`, `sites`,
`component`, `max_rrup`, `record_ids`. Anything more specific is a raw query
through `db.sql(...)` or `db.conn`.

Dimension tables come back whole: `get_events`, `get_realisations`, `get_sites`,
`get_site_event`. Pass `expand_metadata=True` to unpack their JSON `metadata`
column into columns.

## Writing

```python
with IMDB.create("new.duckdb", periods=[0.01, 0.1, 1.0], frequencies=[1.0, 10.0],
                 components=["rotd50"], db_meta={"source": "CyberShake v24p1"}) as db:
    db.add_events(event_df)          # event_id, magnitude, tect_type, ...
    db.add_realisations(rel_df)      # rel_id, event_id, rake, hypo_*, ...
    db.add_sites(site_df)            # site_id, lat, lon, vs30, ...
    db.add_site_event(site_event_df) # site_id, event_id, rrup, rjb, ...
    db.add_records(record_df)        # rel_id, site_id, component, pSA, FAS, PGA, ...
    db.finalise()
```

Integer surrogates (`event_int_id`, `rel_int_id`, `site_int_id`, `record_id`) are
assigned here and are file-local: they change on rebuild, so nothing outside the
file may reference them. Callers work in string IDs throughout.

`add_records` writes `records` plus whichever IM tables the input covers; a row
with no `pSA` array simply gets no `psa_ims` row. `CAV`, `AI`, `Ds575` and
`Ds595` are undefined for `rotd*` components and are stored as NULL there.

Extra source-specific fields go in a `metadata` column, passed as a dict. The
first write of a table's metadata declares its permitted keys in `db_meta`;
later writes are validated against that declaration.

Statements autocommit, so writes are not rolled back on error. To re-ingest an
event, `delete_event(event_id)` first: it removes the event and everything
derived from it, leaving shared sites alone.

## Validation

The large tables carry no `PRIMARY KEY`, `UNIQUE` or `FOREIGN KEY`, because at
these row counts each one is an ART index loaded into memory on open and buys no
lookup speed. `validate()` checks what they would have enforced (orphan keys,
duplicated logical keys, array lengths, undeclared components and metadata keys)
and returns the problems as a list. `finalise()` runs it and raises.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check imdb tests && uv run ruff format --check imdb tests
uv run ty check imdb
```
