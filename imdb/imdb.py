"""Read and write intensity measure databases (IMDBs)."""

import datetime
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any, Self

import ibis
import numpy as np
import pandas as pd
from ibis import _
from ibis.backends.duckdb import Backend as DuckDBBackend

from imdb import schema

logger = logging.getLogger(__name__)


class IMDB:
    """A DuckDB-backed intensity measure database.

    Parameters
    ----------
    path : Path
        Path to the database file.
    read_only : bool
        Open the database read-only.
    """

    def __init__(self, path: Path, read_only: bool = True) -> None:
        """Set up the database path; does not open a connection.

        Parameters
        ----------
        path : Path
            Path to the database file.
        read_only : bool
            Open the database read-only.
        """
        self.path = Path(path)
        self.read_only = read_only
        self._con: DuckDBBackend | None = None

    @property
    def con(self) -> DuckDBBackend:
        """The underlying ibis connection. Raises if the database is not open.

        Returns
        -------
        DuckDBBackend
            The open ibis connection.
        """
        if self._con is None:
            raise RuntimeError(
                "database is not open; call .open() or use as a context manager"
            )
        return self._con

    def open(self) -> Self:
        """Open the database connection, if not already open.

        Returns
        -------
        Self
            This database, open for use.
        """
        if self._con is None:
            self._con = ibis.duckdb.connect(self.path, read_only=self.read_only)
            if "db_meta" in self._con.list_tables():
                found = self.db_meta.get("schema_version")
                if found != schema.SCHEMA_VERSION:
                    self.close()
                    raise RuntimeError(
                        f"database schema version {found!r} does not match "
                        f"this imdb version's {schema.SCHEMA_VERSION!r}; "
                        "the database needs rebuilding with a matching imdb version"
                    )
        return self

    @property
    def db_meta(self) -> dict[str, str]:
        """The `db_meta` table, as a dict.

        Returns
        -------
        dict of str to str
            The `db_meta` table's `key`/`value` rows.
        """
        df = self.con.table("db_meta").to_pandas()
        return dict(zip(df["key"], df["value"], strict=True))

    def close(self) -> None:
        """Close the database connection."""
        if self._con is not None:
            self._con.disconnect()
            self._con = None

    def __enter__(self) -> Self:
        """Open the database connection."""
        return self.open()

    def __exit__(self, *exc: object) -> None:
        """Close the database connection."""
        self.close()

    @classmethod
    def create(
        cls,
        path: Path,
        periods: list[float],
        frequencies: list[float] | None = None,
        components: tuple[str, ...] = schema.COMPONENTS,
        db_meta: dict[str, str] | None = None,
    ) -> "IMDB":
        """Create a new, empty IMDB and return it open for writing.

        Parameters
        ----------
        path : Path
            Path to the database file to create. Must not already exist.
        periods : list of float
            Response spectral periods, in seconds, that `pSA` arrays are indexed by.
        frequencies : list of float, optional
            Frequencies, in Hz, that `FAS` arrays are indexed by.
        components : tuple of str
            Components this database will hold.
        db_meta : dict of str to str, optional
            Extra `db_meta` entries, merged over the defaults.

        Returns
        -------
        IMDB
            The newly created database, open for writing.
        """
        if Path(path).exists():
            raise FileExistsError(f"{path} already exists")
        
        frequencies = frequencies or []
        db = cls(path, read_only=False).open()
        con = db.con
        for statement in schema.DDL.strip().split(";"):
            if statement.strip():
                con.raw_sql(statement)

        con.insert(
            "periods",
            pd.DataFrame(
                {"period_index": range(1, len(periods) + 1), "period": periods}
            ),
        )
        con.insert(
            "frequencies",
            pd.DataFrame(
                {"freq_index": range(1, len(frequencies) + 1), "frequency": frequencies}
            ),
        )
        con.insert(
            "im_units",
            pd.DataFrame(
                {"im": schema.IM_UNITS.keys(), "unit": schema.IM_UNITS.values()}
            ),
        )
        con.insert(
            "notes",
            pd.DataFrame({"topic": schema.NOTES.keys(), "note": schema.NOTES.values()}),
        )

        meta = {
            "schema_version": schema.SCHEMA_VERSION,
            "components": ",".join(components),
            "n_periods": str(len(periods)),
            "n_frequencies": str(len(frequencies)),
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "imdb_version": version("imdb"),
            **(db_meta or {}),
        }
        con.insert(
            "db_meta", pd.DataFrame({"key": meta.keys(), "value": meta.values()})
        )
        return db

    def _next_ids(self, table: str, int_col: str, n: int) -> np.ndarray:
        """Return `n` new contiguous integer ids for `table`, starting after the current max.

        Parameters
        ----------
        table : str
            Table to find the current max id in.
        int_col : str
            The integer id column of `table`.
        n : int
            How many new ids to return.

        Returns
        -------
        np.ndarray
            `n` new contiguous integer ids.
        """
        current = self.con.table(table)[int_col].max().to_pandas()
        start = 0 if pd.isna(current) else int(current) + 1  # ty: ignore[invalid-argument-type]
        return np.arange(start, start + n)

    def _id_map(self, table: str, id_col: str, int_col: str) -> pd.Series:
        """Return a `pd.Series` mapping string id to int id for `table`.

        Parameters
        ----------
        table : str
            Table to read the id mapping from.
        id_col : str
            The stable string id column of `table`.
        int_col : str
            The integer surrogate id column of `table`.

        Returns
        -------
        pd.Series
            Indexed by `id_col`, valued by `int_col`.
        """
        df = self.con.table(table).select(id_col, int_col).to_pandas()
        return df.set_index(id_col)[int_col]

    def add_events(self, df: pd.DataFrame) -> None:
        """Insert new events.

        Parameters
        ----------
        df : pd.DataFrame
            Must have an `event_id` column; other columns match `events`.
        """
        df = df.copy()
        df["event_int_id"] = self._next_ids("events", "event_int_id", len(df))
        self.con.insert("events", df)

    def add_realisations(self, df: pd.DataFrame) -> None:
        """Insert new realisations.

        Parameters
        ----------
        df : pd.DataFrame
            Must have `rel_id` and `event_id` columns; other columns match
            `realisations`.
        """
        df = df.copy()
        event_int_id = self._id_map("events", "event_id", "event_int_id")
        df["event_int_id"] = event_int_id.loc[df["event_id"]].to_numpy()
        df = df.drop(columns="event_id")
        df["rel_int_id"] = self._next_ids("realisations", "rel_int_id", len(df))
        self.con.insert("realisations", df)

    def add_sites(self, df: pd.DataFrame) -> None:
        """Insert new sites.

        Parameters
        ----------
        df : pd.DataFrame
            Must have `site_id`, `lat` and `lon` columns; other columns match `sites`.
        """
        df = df.copy()
        df["site_int_id"] = self._next_ids("sites", "site_int_id", len(df))
        self.con.insert("sites", df)

    def add_site_event(self, df: pd.DataFrame) -> None:
        """Insert new site-event distance rows.

        Parameters
        ----------
        df : pd.DataFrame
            Must have `site_id` and `event_id` columns; other columns match
            `site_event`.
        """
        df = df.copy()
        site_int_id = self._id_map("sites", "site_id", "site_int_id")
        event_int_id = self._id_map("events", "event_id", "event_int_id")
        df["site_int_id"] = site_int_id.loc[df["site_id"]].to_numpy()
        df["event_int_id"] = event_int_id.loc[df["event_id"]].to_numpy()
        df = df.drop(columns=["site_id", "event_id"])
        self.con.insert("site_event", df)

    def add_records(
        self,
        df: pd.DataFrame,
        pSA: np.ndarray | None = None,  # noqa: N803
        FAS: np.ndarray | None = None,  # noqa: N803
        pSA_sigma: np.ndarray | None = None,  # noqa: N803
        FAS_sigma: np.ndarray | None = None,  # noqa: N803
    ) -> np.ndarray:
        """Insert new records, and whichever IM tables the input covers.

        A record whose `pSA`/`FAS` row is entirely NaN is not written to that IM
        table at all, matching the schema's row-presence-means-coverage convention.

        Parameters
        ----------
        df : pd.DataFrame
            Must have `rel_id`, `site_id`, `component` and `kind` columns, one
            row per record. May also have a `gmm_key` column (default `NULL`,
            only meaningful for `kind="gmm"`), any of the scalar IM columns
            (`schema.SCALAR_IMS`), and their paired `<IM>_sigma` columns.
        pSA : np.ndarray, optional
            Shape `(len(df), n_periods)`. A row of all NaN means that record has
            no pSA.
        FAS : np.ndarray, optional
            Shape `(len(df), n_frequencies)`. A row of all NaN means that record
            has no FAS.
        pSA_sigma : np.ndarray, optional
            Shape `(len(df), n_periods)`, ln-space total sigma paired with `pSA`.
            Only valid together with `pSA`.
        FAS_sigma : np.ndarray, optional
            Shape `(len(df), n_frequencies)`, ln-space total sigma paired with
            `FAS`. Only valid together with `FAS`.

        Returns
        -------
        np.ndarray
            The `record_int_id` assigned to each row of `df`, in input order.

        Raises
        ------
        ValueError
            If `df` contains duplicate `(rel_id, site_id, component, kind, gmm_key)`
            rows, or any of them already exist in this database.
        """
        df = df.copy()

        if "kind" not in df:
            raise ValueError(
                "df must have a 'kind' column ('simulated', 'gmm' or 'observed')"
            )
        if "gmm_key" not in df:
            df["gmm_key"] = None

        unknown = set(df["component"]) - set(self.db_meta["components"].split(","))
        if unknown:
            raise ValueError(f"components not in this database: {sorted(unknown)}")

        if pSA is not None and pSA.shape[0] != len(df):
            raise ValueError("pSA must have one row per record")
        if FAS is not None and FAS.shape[0] != len(df):
            raise ValueError("FAS must have one row per record")
        if pSA_sigma is not None and not (
            pSA is not None and pSA_sigma.shape == pSA.shape
        ):
            raise ValueError(
                "pSA_sigma must match pSA's shape, and only be given together with pSA"
            )
        if FAS_sigma is not None and not (
            FAS is not None and FAS_sigma.shape == FAS.shape
        ):
            raise ValueError(
                "FAS_sigma must match FAS's shape, and only be given together with FAS"
            )

        rel_int_id_mapping = self._id_map("realisations", "rel_id", "rel_int_id")
        rel_event_int_id_mapping = self._id_map(
            "realisations", "rel_int_id", "event_int_id"
        )
        site_int_id_mapping = self._id_map("sites", "site_id", "site_int_id")

        df["rel_int_id"] = rel_int_id_mapping.loc[df["rel_id"]].to_numpy()
        df["site_int_id"] = site_int_id_mapping.loc[df["site_id"]].to_numpy()
        df["event_int_id"] = rel_event_int_id_mapping.loc[df["rel_int_id"]].to_numpy()

        key_cols = ["rel_int_id", "site_int_id", "component", "kind", "gmm_key"]
        keys = df[key_cols]
        if keys.duplicated().any():
            raise ValueError(
                "df contains duplicate records (same rel_id, site_id, component, "
                "kind and gmm_key)"
            )
        existing = (
            self.con.table("records")
            .filter(_.rel_int_id.isin(df["rel_int_id"].unique()))
            .select(*key_cols)
            .to_pandas()
        )
        collisions = keys.merge(existing, on=key_cols, how="inner")
        if not collisions.empty:
            raise ValueError(
                f"{len(collisions)} record(s) already exist in this database "
                "(same rel_id, site_id, component, kind and gmm_key)"
            )

        record_int_id = (
            self.con.raw_sql(
                f"SELECT nextval('record_int_id_seq') FROM range({len(df)})"
            )
            .df()["nextval('record_int_id_seq')"]
            .to_numpy()
        )
        df["record_int_id"] = record_int_id

        self.con.raw_sql("BEGIN TRANSACTION")
        try:
            self.con.insert(
                "records",
                df[
                    [
                        "record_int_id",
                        "event_int_id",
                        "rel_int_id",
                        "site_int_id",
                        "component",
                        "kind",
                        "gmm_key",
                    ]
                ],
            )
            if pSA is not None:
                mask = ~np.isnan(pSA).all(axis=1)
                psa_df = pd.DataFrame(
                    {"record_int_id": record_int_id[mask], "pSA": list(pSA[mask])}
                )
                psa_df["pSA_sigma"] = (
                    list(pSA_sigma[mask]) if pSA_sigma is not None else None
                )
                self.con.insert("psa_ims", psa_df)
            if FAS is not None:
                mask = ~np.isnan(FAS).all(axis=1)
                fas_df = pd.DataFrame(
                    {"record_int_id": record_int_id[mask], "FAS": list(FAS[mask])}
                )
                fas_df["FAS_sigma"] = (
                    list(FAS_sigma[mask]) if FAS_sigma is not None else None
                )
                self.con.insert("fas_ims", fas_df)
            scalar_cols = [c for c in schema.SCALAR_IMS if c in df]
            sigma_cols = [f"{c}_sigma" for c in scalar_cols if f"{c}_sigma" in df]
            if scalar_cols:
                scalars = df[["record_int_id", *scalar_cols, *sigma_cols]].copy()
                rotd = df["component"].str.startswith("rotd")
                for col in schema.ROTD_UNDEFINED & set(scalar_cols):
                    scalars.loc[rotd, col] = None
                    if f"{col}_sigma" in scalars:
                        scalars.loc[rotd, f"{col}_sigma"] = None
                self.con.insert("scalars_ims", scalars)
        except Exception:
            self.con.raw_sql("ROLLBACK")
            raise
        self.con.raw_sql("COMMIT")
        logger.info("inserted %d records", len(df))
        return record_int_id

    def delete_event(self, event_id: str) -> None:
        """Delete an event and everything derived from it, for a clean re-ingest.

        Parameters
        ----------
        event_id : str
            The event to delete.
        """
        event_int_id_subquery = "(SELECT event_int_id FROM events WHERE event_id = ?)"
        record_subquery = f"(SELECT record_int_id FROM records WHERE event_int_id = {event_int_id_subquery})"
        statements = [
            f"DELETE FROM psa_ims WHERE record_int_id IN {record_subquery}",
            f"DELETE FROM fas_ims WHERE record_int_id IN {record_subquery}",
            f"DELETE FROM scalars_ims WHERE record_int_id IN {record_subquery}",
            f"DELETE FROM records WHERE event_int_id = {event_int_id_subquery}",
            f"DELETE FROM site_event WHERE event_int_id = {event_int_id_subquery}",
            f"DELETE FROM realisations WHERE event_int_id = {event_int_id_subquery}",
            ##  QUESTION:Why is this one a different style than the other deletes? Why not f string?
            "DELETE FROM events WHERE event_id = ?",
        ]
        for statement in statements:
            self.con.raw_sql(statement, parameters=[event_id])

    def validate(self) -> list[str]:
        """Check the invariants the schema itself cannot enforce: unique ids, valid FKs.

        Returns
        -------
        list of str
            One entry per problem found; empty if the database is consistent.
        """
        problems = []
        for table in ("records", "psa_ims", "fas_ims", "scalars_ims"):
            t = self.con.table(table)
            n_rows = t.count().to_pandas()
            n_unique = t.record_int_id.nunique().to_pandas()
            if n_rows != n_unique:
                problems.append(
                    f"{table}: record_int_id is not unique ({n_rows} rows, {n_unique} unique)"
                )

        records = self.con.table("records")
        fks = {
            "rel_int_id": self.con.table("realisations").rel_int_id,
            "site_int_id": self.con.table("sites").site_int_id,
            "event_int_id": self.con.table("events").event_int_id,
        }
        for col, valid in fks.items():
            n = records.filter(~records[col].isin(valid)).count().to_pandas()
            if n:
                problems.append(f"records: {n} rows with an unknown {col}")

        for table in ("psa_ims", "fas_ims", "scalars_ims"):
            t = self.con.table(table)
            n = (
                t.filter(~t.record_int_id.isin(records.record_int_id))
                .count()
                .to_pandas()
            )
            if n:
                problems.append(f"{table}: {n} rows with an unknown record_int_id")

        bad_gmm_key = (
            records.filter(
                ((records.kind == "gmm") & records.gmm_key.isnull())
                | ((records.kind != "gmm") & records.gmm_key.notnull())
            )
            .count()
            .to_pandas()
        )
        if bad_gmm_key:
            problems.append(
                f"records: {bad_gmm_key} rows with gmm_key inconsistent with kind"
            )

        kind_by_record = records.select("record_int_id", "kind")
        sigma_cols = [
            ("psa_ims", "pSA_sigma"),
            ("fas_ims", "FAS_sigma"),
            *[("scalars_ims", f"{im}_sigma") for im in schema.SCALAR_IMS],
        ]
        for table, col in sigma_cols:
            joined = self.con.table(table).join(kind_by_record, "record_int_id")
            n = (
                joined.filter((joined.kind != "gmm") & joined[col].notnull())
                .count()
                .to_pandas()
            )
            if n:
                problems.append(f"{table}.{col}: {n} rows populated for kind != 'gmm'")

        return problems

    # ---- read -------------------------------------------------------------

    def get_events(self) -> pd.DataFrame:
        """Return all events, indexed by `event_id`.

        Returns
        -------
        pd.DataFrame
            One row per event.
        """
        return self.con.table("events").to_pandas().set_index("event_id")

    def get_realisations(self) -> pd.DataFrame:
        """Return all realisations, indexed by `rel_id`.

        Returns
        -------
        pd.DataFrame
            One row per realisation.
        """
        return self.con.table("realisations").to_pandas().set_index("rel_id")

    def get_sites(self) -> pd.DataFrame:
        """Return all sites, indexed by `site_id`.

        Returns
        -------
        pd.DataFrame
            One row per site.
        """
        return self.con.table("sites").to_pandas().set_index("site_id")

    def get_site_event(
        self,
        event_ids: list[str] | None = None,
        site_ids: list[str] | None = None,
        max_rrup: float | None = None,
    ) -> pd.DataFrame:
        """Return site-event distance rows.

        Parameters
        ----------
        event_ids : list of str, optional
            Only these events.
        site_ids : list of str, optional
            Only these sites.
        max_rrup : float, optional
            Only rows with `rrup` at most this value.

        Returns
        -------
        pd.DataFrame
            One row per (site, event), with `site_id` and `event_id` columns.
        """
        sites = self.con.table("sites").select("site_int_id", "site_id")
        events = self.con.table("events").select("event_int_id", "event_id")
        t = (
            self.con.table("site_event")
            .join(sites, "site_int_id")
            .join(events, "event_int_id")
        )
        if event_ids is not None:
            t = t.filter(t.event_id.isin(event_ids))
        if site_ids is not None:
            t = t.filter(t.site_id.isin(site_ids))
        if max_rrup is not None:
            t = t.filter(t.rrup <= max_rrup)
        return t.drop("site_int_id", "event_int_id").to_pandas()

    def get_records(
        self,
        event_ids: list[str] | None = None,
        rel_ids: list[str] | None = None,
        site_ids: list[str] | None = None,
        component: str | None = None,
        kind: str | None = None,
        gmm_key: str | None = None,
        record_int_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Return record identity rows.

        Parameters
        ----------
        event_ids : list of str, optional
            Only records for these events.
        rel_ids : list of str, optional
            Only records for these realisations.
        site_ids : list of str, optional
            Only records for these sites.
        component : str, optional
            Only records with this component.
        kind : str, optional
            Only records of this kind (`"simulated"`, `"gmm"` or `"observed"`).
        gmm_key : str, optional
            Only records from this GMM.
        record_int_ids : list of int, optional
            Only these `record_int_id` values.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_int_id`, with `event_id`, `rel_id`, `site_id`,
            `component`, `kind` and `gmm_key` columns.
        """
        events = self.con.table("events").select("event_int_id", "event_id")
        realisations = self.con.table("realisations").select("rel_int_id", "rel_id")
        sites = self.con.table("sites").select("site_int_id", "site_id")
        t = (
            self.con.table("records")
            .join(events, "event_int_id")
            .join(realisations, "rel_int_id")
            .join(sites, "site_int_id")
        )
        if event_ids is not None:
            t = t.filter(t.event_id.isin(event_ids))
        if rel_ids is not None:
            t = t.filter(t.rel_id.isin(rel_ids))
        if site_ids is not None:
            t = t.filter(t.site_id.isin(site_ids))
        if component is not None:
            t = t.filter(t.component == component)
        if kind is not None:
            t = t.filter(t.kind == kind)
        if gmm_key is not None:
            t = t.filter(t.gmm_key == gmm_key)
        if record_int_ids is not None:
            ids = ibis.memtable({"record_int_id": list(record_int_ids)})
            t = t.semi_join(ids, "record_int_id")
        return (
            t.select(
                "record_int_id",
                "event_id",
                "rel_id",
                "site_id",
                "component",
                "kind",
                "gmm_key",
            )
            .to_pandas()
            .set_index("record_int_id")
        )

    def get_psa(
        self,
        periods: list[float] | None = None,
        sigma: bool = False,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return response spectral acceleration.

        Parameters
        ----------
        periods : list of float, optional
            Periods to return, in seconds. Defaults to every period on the grid.
        sigma : bool
            Also return each period's ln-space total sigma, as `<period>_sigma`
            columns.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_int_id`, one column per requested period.
        """
        period_index = (
            self.con.table("periods").to_pandas().set_index("period")["period_index"]
        )
        if periods is None:
            periods = period_index.index.tolist()
        record_int_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("psa_ims").filter(_.record_int_id.isin(record_int_ids))
        cols = {str(p): t.pSA[int(period_index.loc[p]) - 1] for p in periods}
        if sigma:
            cols.update(
                {
                    f"{p}_sigma": t.pSA_sigma[int(period_index.loc[p]) - 1]
                    for p in periods
                }
            )
        return (
            t.select(record_int_id=t.record_int_id, **cols)
            .to_pandas()
            .set_index("record_int_id")
        )

    def get_fas(
        self,
        frequencies: list[float] | None = None,
        sigma: bool = False,
        **filters: Any,
    ) -> pd.DataFrame:
        """Return Fourier amplitude spectra.

        Parameters
        ----------
        frequencies : list of float, optional
            Frequencies to return, in Hz. Defaults to every frequency on the grid.
        sigma : bool
            Also return each frequency's ln-space total sigma, as `<frequency>_sigma`
            columns.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_int_id`, one column per requested frequency.
        """
        freq_index = (
            self.con.table("frequencies")
            .to_pandas()
            .set_index("frequency")["freq_index"]
        )
        if frequencies is None:
            frequencies = freq_index.index.tolist()
        record_int_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("fas_ims").filter(_.record_int_id.isin(record_int_ids))
        cols = {str(f): t.FAS[int(freq_index.loc[f]) - 1] for f in frequencies}
        if sigma:
            cols.update(
                {
                    f"{f}_sigma": t.FAS_sigma[int(freq_index.loc[f]) - 1]
                    for f in frequencies
                }
            )
        return (
            t.select(record_int_id=t.record_int_id, **cols)
            .to_pandas()
            .set_index("record_int_id")
        )

    def get_scalars(
        self, ims: list[str] | None = None, sigma: bool = False, **filters: Any
    ) -> pd.DataFrame:
        """Return scalar intensity measures.

        Parameters
        ----------
        ims : list of str, optional
            Which scalar IMs to return. Defaults to all of `schema.SCALAR_IMS`.
        sigma : bool
            Also return each IM's ln-space total sigma, as `<IM>_sigma` columns.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_int_id`, one column per requested IM.
        """
        ims = ims or list(schema.SCALAR_IMS)
        cols = [*ims, *([f"{im}_sigma" for im in ims] if sigma else [])]
        record_int_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("scalars_ims").filter(_.record_int_id.isin(record_int_ids))
        return t.select("record_int_id", *cols).to_pandas().set_index("record_int_id")
