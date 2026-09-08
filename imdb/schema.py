"""
Schema definition for the intensity measure database.

This module holds facts only: the DDL, the fixed vocabularies and the
documentation text written into every database. The DDL is the single source
of truth for table and column names; nothing else in the package hardcodes a
column list.
"""

SCHEMA_VERSION = "0"

COMPONENTS = ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")
"""Ground-motion components, following ``IM_calculation``."""

TECT_TYPES = (
    "ACTIVE_SHALLOW",
    "VOLCANIC",
    "SUBDUCTION_INTERFACE",
    "SUBDUCTION_SLAB",
)
"""Tectonic types, following the ``qcore``/``workflow`` ``TectType`` vocabulary."""

SCALAR_IMS = ("PGA", "PGV", "PGD", "CAV", "AI", "Ds575", "Ds595")
"""Scalar intensity measures, in ``scalars_ims`` column order."""

ROTD_UNDEFINED = frozenset({"CAV", "AI", "Ds575", "Ds595"})
"""Scalar IMs that are undefined for ``rotd*`` components and stored as NULL."""

IM_UNITS = {
    "pSA": "g",
    "FAS": "g.s",
    "PGA": "g",
    "PGV": "cm/s",
    "PGD": "cm",
    "CAV": "m/s",
    "AI": "m/s",
    "Ds575": "s",
    "Ds595": "s",
}
"""Linear physical unit of each intensity measure. Log is a read-time transform."""

METADATA_TABLES = {
    "events": "event_metadata_keys",
    "realisations": "rel_metadata_keys",
    "sites": "site_metadata_keys",
    "site_event": "site_event_metadata_keys",
}
"""Tables carrying a JSON ``metadata`` column, and the ``db_meta`` key declaring its keys."""

IM_TABLES = {
    "psa_ims": "pSA",
    "fas_ims": "FAS",
    "scalars_ims": None,
}
"""IM tables, mapped to their array column where they have one."""

NOTES = {
    "logical keys": (
        "site_event is keyed on (site_int_id, event_int_id); records on "
        "(rel_int_id, site_int_id, component); each IM table on record_id. None of "
        "these are declared as constraints. They are enforced by the writer and "
        "checked by IMDB.validate()."
    ),
    "array indexing is 1-based": (
        "periods.period_index and frequencies.freq_index are 1-based, so pSA[period_index] "
        "and FAS[freq_index] need no offset. len(pSA) equals the row count of periods and "
        "len(FAS) the row count of frequencies, for every row."
    ),
    "identity and rebuild stability": (
        "event_id, rel_id and site_id are stable. The integer surrogates event_int_id, "
        "rel_int_id, site_int_id and record_id are assigned at ingest and change on "
        "rebuild. Nothing outside this database may reference them."
    ),
    "component vocabulary": (
        "000, 090, ver, geom, rotd0, rotd50, rotd100, following IM_calculation. This "
        "database holds the subset listed in db_meta.components."
    ),
    "rotd scalars are undefined": (
        "CAV, AI, Ds575 and Ds595 are undefined for rotd0, rotd50 and rotd100 and are "
        "stored as NULL for those components. PGA, PGV and PGD are populated for every "
        "component."
    ),
    "units": (
        "IM units are one row each in im_units, and are linear physical units. Elsewhere: "
        "distances km, vs30 m/s, z1p0 and z2p5 km, depths km, angles degrees, coordinates "
        "WGS84."
    ),
    "metadata columns are JSON": (
        "events, realisations, sites and site_event each carry a metadata VARCHAR holding "
        "a JSON object. Read with json_extract_string(metadata, '$.key'). Permitted keys "
        "are declared in db_meta. A predicate on a metadata field cannot use zone-map "
        "pruning."
    ),
    "distances are event level": (
        "rrup, rjb, rx and ry are measured to the rupture surface and are shared across "
        "all realisations of an event. Hypocentral and epicentral distance are not stored; "
        "compute them from realisations.hypo_* and sites.lat/lon."
    ),
    "synthetic realisations": (
        "Every event has at least one realisation. A dataset with no realisation concept "
        "gets exactly one per event, with rel_id equal to event_id."
    ),
    "record_id is file-local": (
        "record_id comes from a sequence and shifts on rebuild. External references must "
        "cite (rel_id, site_id, component)."
    ),
    "physical sort order": (
        "Rows are written in the order recorded by db_meta.sort_order. Filters on the sort "
        "key prune row groups; filters on anything else do not."
    ),
    "provenance": (
        "db_meta records who built this database, when, from what source, and with which "
        "version of the imdb library."
    ),
}
"""Self-documenting notes written into the ``notes`` table by :meth:`imdb.IMDB.create`."""

DDL = """
-- ---------- documentation ----------

CREATE TABLE db_meta  (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL);
CREATE TABLE notes    (topic VARCHAR PRIMARY KEY, note VARCHAR NOT NULL);
CREATE TABLE im_units (im VARCHAR PRIMARY KEY, unit VARCHAR NOT NULL);

-- ---------- IM vocabulary (1-based, matching DuckDB list indexing) ----------

CREATE TABLE periods (
    period_index INTEGER PRIMARY KEY,
    period       DOUBLE  NOT NULL UNIQUE       -- seconds
);

CREATE TABLE frequencies (
    freq_index   INTEGER PRIMARY KEY,
    frequency    DOUBLE  NOT NULL UNIQUE       -- Hz
);

-- ---------- dimensions ----------

CREATE TYPE tect_type_t AS ENUM (
    'ACTIVE_SHALLOW', 'VOLCANIC', 'SUBDUCTION_INTERFACE', 'SUBDUCTION_SLAB'
);

CREATE TABLE events (
    event_int_id INTEGER PRIMARY KEY,
    event_id     VARCHAR NOT NULL UNIQUE,
    magnitude    FLOAT,
    tect_type    tect_type_t,
    dip          FLOAT,
    dip_dir      FLOAT,
    dtop         FLOAT,
    dbottom      FLOAT,
    length       FLOAT,
    source_wkt   VARCHAR,        -- rupture surface
    trace_wkt    VARCHAR,        -- surface trace
    domain_wkt   VARCHAR,        -- simulation domain
    metadata     VARCHAR         -- JSON: fault_type, sim_type, plane_count, ...
);

CREATE TABLE realisations (
    rel_int_id   INTEGER PRIMARY KEY,
    rel_id       VARCHAR NOT NULL UNIQUE,
    event_int_id INTEGER NOT NULL REFERENCES events(event_int_id),
    magnitude    FLOAT,
    rake         FLOAT,
    hypo_lat     FLOAT,
    hypo_lon     FLOAT,
    hypo_depth   FLOAT,
    metadata     VARCHAR         -- JSON: solver, shypo, dhypo, ...
);

CREATE TABLE sites (
    site_int_id INTEGER PRIMARY KEY,
    site_id     VARCHAR NOT NULL UNIQUE,
    lat         FLOAT NOT NULL,
    lon         FLOAT NOT NULL,
    vs30        FLOAT,           -- m/s
    z1p0        FLOAT,           -- km
    z2p5        FLOAT,           -- km
    metadata    VARCHAR          -- JSON: elevation, basin, grid_level, ...
);

-- ---------- large tables: no PRIMARY KEY, UNIQUE or FOREIGN KEY ----------

CREATE TABLE site_event (
    site_int_id  INTEGER NOT NULL,
    event_int_id INTEGER NOT NULL,
    rrup         FLOAT,          -- km
    rjb          FLOAT,
    rx           FLOAT,
    ry           FLOAT,
    metadata     VARCHAR         -- JSON
);
-- logical key (site_int_id, event_int_id)

CREATE SEQUENCE record_id_seq START 1;

CREATE TABLE records (
    record_id    BIGINT DEFAULT nextval('record_id_seq'),
    event_int_id INTEGER NOT NULL,   -- derived from rel_int_id by the writer
    rel_int_id   INTEGER NOT NULL,
    site_int_id  INTEGER NOT NULL,
    component    VARCHAR NOT NULL
);
-- logical key (rel_int_id, site_int_id, component)

CREATE TABLE psa_ims (record_id BIGINT NOT NULL, pSA FLOAT[]);  -- periods.period_index
CREATE TABLE fas_ims (record_id BIGINT NOT NULL, FAS FLOAT[]);  -- frequencies.freq_index

CREATE TABLE scalars_ims (
    record_id BIGINT NOT NULL,
    PGA       FLOAT,
    PGV       FLOAT,
    PGD       FLOAT,
    CAV       FLOAT,   -- NULL for rotd components
    AI        FLOAT,   -- NULL for rotd components
    Ds575     FLOAT,   -- NULL for rotd components
    Ds595     FLOAT    -- NULL for rotd components
);
"""
"""Full schema DDL, executed as one script by :meth:`imdb.IMDB.create`."""
