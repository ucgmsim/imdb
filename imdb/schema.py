"""DDL and fixed vocabulary for the IMDB schema, version 0.

Facts only, no logic. `DDL` is the single source of truth for the schema; nothing
else in this library composes column lists by hand.
"""

SCHEMA_VERSION = "0"

DDL = """
CREATE TABLE db_meta  (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL);
CREATE TABLE notes    (topic VARCHAR PRIMARY KEY, note VARCHAR NOT NULL);
CREATE TABLE im_units (im VARCHAR PRIMARY KEY, unit VARCHAR NOT NULL);

CREATE TABLE periods (
    period_index INTEGER PRIMARY KEY,
    period       DOUBLE  NOT NULL UNIQUE
);

CREATE TABLE frequencies (
    freq_index   INTEGER PRIMARY KEY,
    frequency    DOUBLE  NOT NULL UNIQUE
);

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
    source_wkt   VARCHAR,
    trace_wkt    VARCHAR,
    domain_wkt   VARCHAR,
    metadata     VARCHAR
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
    metadata     VARCHAR
);

CREATE TABLE sites (
    site_int_id INTEGER PRIMARY KEY,
    site_id     VARCHAR NOT NULL UNIQUE,
    lat         FLOAT NOT NULL,
    lon         FLOAT NOT NULL,
    vs30        FLOAT,
    z1p0        FLOAT,
    z2p5        FLOAT,
    metadata    VARCHAR
);

CREATE TABLE site_event (
    site_int_id  INTEGER NOT NULL,
    event_int_id INTEGER NOT NULL,
    rrup         FLOAT,
    rjb          FLOAT,
    rx           FLOAT,
    ry           FLOAT,
    metadata     VARCHAR
);

CREATE SEQUENCE record_id_seq START 1;

CREATE TABLE records (
    record_id    BIGINT DEFAULT nextval('record_id_seq'),
    event_int_id INTEGER NOT NULL,
    rel_int_id   INTEGER NOT NULL,
    site_int_id  INTEGER NOT NULL,
    component    VARCHAR NOT NULL
);

CREATE TABLE psa_ims (record_id BIGINT NOT NULL, pSA FLOAT[]);
CREATE TABLE fas_ims (record_id BIGINT NOT NULL, FAS FLOAT[]);

CREATE TABLE scalars_ims (
    record_id BIGINT NOT NULL,
    PGA       FLOAT,
    PGV       FLOAT,
    PGD       FLOAT,
    CAV       FLOAT,
    AI        FLOAT,
    Ds575     FLOAT,
    Ds595     FLOAT
);
"""

COMPONENTS = ("000", "090", "ver", "geom", "rotd0", "rotd50", "rotd100")

SCALAR_IMS = ("PGA", "PGV", "PGD", "CAV", "AI", "Ds575", "Ds595")

ROTD_UNDEFINED = frozenset({"CAV", "AI", "Ds575", "Ds595"})

TECT_TYPES = ("ACTIVE_SHALLOW", "VOLCANIC", "SUBDUCTION_INTERFACE", "SUBDUCTION_SLAB")

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

METADATA_TABLES = {
    "events": "event_metadata_keys",
    "realisations": "rel_metadata_keys",
    "sites": "site_metadata_keys",
    "site_event": "site_event_metadata_keys",
}

NOTES = {
    "logical keys": (
        "site_event's logical key is (site_int_id, event_int_id); records' logical key is "
        "(rel_int_id, site_int_id, component). Neither is enforced by a constraint; the "
        "writer is responsible for not creating duplicates."
    ),
    "array indexing is 1-based": (
        "periods.period_index and frequencies.freq_index are 1-based, matching DuckDB list "
        "indexing, so pSA[period_index] and FAS[freq_index] need no offset. len(pSA) equals "
        "the row count of periods; len(FAS) equals the row count of frequencies."
    ),
    "identity and rebuild stability": (
        "event_id, rel_id and site_id are stable. The integer surrogates event_int_id, "
        "rel_int_id, site_int_id and record_id are assigned at ingest and change on rebuild; "
        "nothing outside the database may reference them. External references use "
        "(rel_id, site_id, component)."
    ),
    "component vocabulary": (
        "000, 090, ver, geom, rotd0, rotd50, rotd100, following IM_calculation. A database "
        "may hold any subset; db_meta.components lists which. The writer validates against "
        "that list."
    ),
    "rotd scalars are undefined": (
        "scalars_ims.CAV, AI, Ds575 and Ds595 are NULL for rotd* components. PGA, PGV and "
        "PGD are populated for every component."
    ),
    "units": (
        "Linear, physical units; log is a read-time transform. Every IM unit is a row in "
        "im_units. Outside the IM tables: distances km, vs30 m/s, z1p0 and z2p5 km, depths "
        "km, angles degrees, coordinates WGS84."
    ),
    "metadata columns are JSON": (
        "events, realisations, sites and site_event each carry a metadata VARCHAR holding a "
        "JSON object. Read with json_extract_string(metadata, '$.key'). Permitted keys are "
        "declared in db_meta and validated by the writer."
    ),
    "distances are event level": (
        "rrup, rjb, rx and ry are measured to the rupture surface and are defined at the "
        "event level, shared across all realisations of an event. Hypocentral and "
        "epicentral distance are not stored; compute them from realisations.hypo_* and "
        "sites.lat/lon."
    ),
    "synthetic realisations": (
        "Every event has at least one realisation. A dataset with no realisation concept "
        "gets exactly one per event, with rel_id equal to event_id."
    ),
    "record_id is file-local": (
        "record_id is a surrogate scoped to this database file only; it is not stable "
        "across rebuilds or between databases."
    ),
    "physical sort order": (
        "Rows are written in event order, so event_int_id and record_id are monotonic and "
        "event filters prune row groups. Site filters do not prune. db_meta.sort_order "
        "records the ordering used."
    ),
    "provenance": (
        "db_meta.source, creator, created_at and imdb_version record where the data in this "
        "database came from and what built it."
    ),
}
