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
        """Set up the database path; does not open a connection."""
        self.path = Path(path)
        self.read_only = read_only
        self._con: DuckDBBackend | None = None

    @property
    def con(self) -> DuckDBBackend:
        """The underlying ibis connection. Raises if the database is not open."""
        if self._con is None:
            raise RuntimeError(
                "database is not open; call .open() or use as a context manager"
            )
        return self._con

    def open(self) -> Self:
        """Open the database connection, if not already open."""
        if self._con is None:
            self._con = ibis.duckdb.connect(self.path, read_only=self.read_only)
        return self

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

    # ---- create -------------------------------------------------------

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

    # ---- write helpers --------------------------------------------------

    def _next_ids(self, table: str, int_col: str, n: int) -> np.ndarray:
        """Return `n` new contiguous integer ids for `table`, starting after the current max."""
        current = self.con.table(table)[int_col].max().to_pandas()
        start = 0 if pd.isna(current) else int(current) + 1  # ty: ignore[invalid-argument-type]
        return np.arange(start, start + n)

    def _id_map(self, table: str, id_col: str, int_col: str) -> pd.Series:
        """Return a `pd.Series` mapping string id to int id for `table`."""
        df = self.con.table(table).select(id_col, int_col).to_pandas()
        return df.set_index(id_col)[int_col]

    # ---- write ----------------------------------------------------------

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

    ## TODO: CHange this to take a record_df (containing rel_id, site_id, component) + scalar IM columns, 
    # plus optional pSA and FAS numpy arrays. Update logic accordingly
    def add_records(self, df: pd.DataFrame) -> np.ndarray:
        """Insert new records, and whichever IM tables the input covers.

        A row with a missing (`None`) `pSA` or `FAS` array is not written to that IM
        table at all, matching the schema's row-presence-means-coverage convention.

        Parameters
        ----------
        df : pd.DataFrame
            Must have `rel_id`, `site_id` and `component` columns. May also have a
            `pSA` column (list of float, one per period), a `FAS` column (list of
            float, one per frequency), and any of the scalar IM columns
            (`schema.SCALAR_IMS`).

        Returns
        -------
        np.ndarray
            The `record_id` assigned to each input row, in input order.
        """
        df = df.copy()

        ## TODO: I think this should be a property?
        components = set(
            self.con.table("db_meta")
            .filter(_.key == "components")
            .to_pandas()["value"]
            .iloc[0]
            .split(",")
        )
        unknown = set(df["component"]) - components
        if unknown:
            raise ValueError(f"components not in this database: {sorted(unknown)}")

        rel_int_id = self._id_map("realisations", "rel_id", "rel_int_id")
        rel_event_int_id = self._id_map("realisations", "rel_int_id", "event_int_id")
        site_int_id = self._id_map("sites", "site_id", "site_int_id")

        df["rel_int_id"] = rel_int_id.loc[df["rel_id"]].to_numpy()
        df["site_int_id"] = site_int_id.loc[df["site_id"]].to_numpy()
        df["event_int_id"] = rel_event_int_id.loc[df["rel_int_id"]].to_numpy()
        record_id = (
            self.con.raw_sql(f"SELECT nextval('record_id_seq') FROM range({len(df)})")
            .df()["nextval('record_id_seq')"]
            .to_numpy()
        )
        df["record_id"] = record_id

        self.con.raw_sql("BEGIN TRANSACTION")
        try:
            self.con.insert(
                "records",
                df[
                    [
                        "record_id",
                        "event_int_id",
                        "rel_int_id",
                        "site_int_id",
                        "component",
                    ]
                ],
            )
            if "pSA" in df:
                # Why not just 
                mask = df["pSA"].apply(lambda x: x is not None)
                self.con.insert("psa_ims", df.loc[mask, ["record_id", "pSA"]])
            if "FAS" in df:
                mask = df["FAS"].apply(lambda x: x is not None)
                self.con.insert("fas_ims", df.loc[mask, ["record_id", "FAS"]])
            scalar_cols = [c for c in schema.SCALAR_IMS if c in df]
            if scalar_cols:
                scalars = df[["record_id", *scalar_cols]].copy()
                rotd = df["component"].str.startswith("rotd")
                for col in schema.ROTD_UNDEFINED & set(scalar_cols):
                    scalars.loc[rotd, col] = None
                self.con.insert("scalars_ims", scalars)
        except Exception:
            self.con.raw_sql("ROLLBACK")
            raise
        self.con.raw_sql("COMMIT")
        logger.info("inserted %d records", len(df))
        return record_id

    def delete_event(self, event_id: str) -> None:
        """Delete an event and everything derived from it, for a clean re-ingest.

        Parameters
        ----------
        event_id : str
            The event to delete.
        """
        event_int_id_subquery = "(SELECT event_int_id FROM events WHERE event_id = ?)"
        record_subquery = f"(SELECT record_id FROM records WHERE event_int_id = {event_int_id_subquery})"
        statements = [
            f"DELETE FROM psa_ims WHERE record_id IN {record_subquery}",
            f"DELETE FROM fas_ims WHERE record_id IN {record_subquery}",
            f"DELETE FROM scalars_ims WHERE record_id IN {record_subquery}",
            f"DELETE FROM records WHERE event_int_id = {event_int_id_subquery}",
            f"DELETE FROM site_event WHERE event_int_id = {event_int_id_subquery}",
            f"DELETE FROM realisations WHERE event_int_id = {event_int_id_subquery}",
        ]
        for statement in statements:
            self.con.raw_sql(statement, parameters=[event_id])

    ## TODO: Simplify, it should just check for basic stuff, e.g. that all integer ids are unique, fks are valid. Keep it simple.
    def validate(self) -> list[str]:
        """Check the invariants the schema itself cannot enforce.

        Returns
        -------
        list of str
            One entry per problem found; empty if the database is consistent.
        """
        problems = []
        records = self.con.table("records")
        n_periods = len(self.con.table("periods").to_pandas())
        n_frequencies = len(self.con.table("frequencies").to_pandas())

        orphans = {
            "rel_int_id": self.con.table("realisations").rel_int_id,
            "site_int_id": self.con.table("sites").site_int_id,
            "event_int_id": self.con.table("events").event_int_id,
        }
        for col, valid in orphans.items():
            n = records.filter(~records[col].isin(valid)).count().to_pandas()
            if n:
                problems.append(f"records: {n} rows with an unknown {col}")

        n = (
            self.con.table("psa_ims")
            .filter(_.pSA.length() != n_periods)
            .count()
            .to_pandas()
        )
        if n:
            problems.append(f"psa_ims: {n} rows with pSA length != {n_periods}")

        n = (
            self.con.table("fas_ims")
            .filter(_.FAS.length() != n_frequencies)
            .count()
            .to_pandas()
        )
        if n:
            problems.append(f"fas_ims: {n} rows with FAS length != {n_frequencies}")

        return problems

    # ---- read -------------------------------------------------------------

    def get_events(self) -> pd.DataFrame:
        """Return all events, indexed by `event_id`."""
        return self.con.table("events").to_pandas().set_index("event_id")

    def get_realisations(self) -> pd.DataFrame:
        """Return all realisations, indexed by `rel_id`."""
        return self.con.table("realisations").to_pandas().set_index("rel_id")

    def get_sites(self) -> pd.DataFrame:
        """Return all sites, indexed by `site_id`."""
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
        record_ids: list[int] | None = None,
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
        record_ids : list of int, optional
            Only these `record_id` values.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_id`, with `event_id`, `rel_id`, `site_id` and
            `component` columns.
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
        if record_ids is not None:
            ## QUESTION: How performant is this for a large number of record_ids? I previously had issues with ISIN (SQL) queries being slow.
            t = t.filter(t.record_id.isin(record_ids))
        return (
            t.select("record_id", "event_id", "rel_id", "site_id", "component")
            .to_pandas()
            .set_index("record_id")
        )

    def get_psa(
        self, periods: list[float] | None = None, **filters: Any
    ) -> pd.DataFrame:
        """Return response spectral acceleration.

        Parameters
        ----------
        periods : list of float, optional
            Periods to return, in seconds. Defaults to every period on the grid.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_id`, one column per requested period.
        """
        grid = self.con.table("periods").to_pandas().set_index("period")["period_index"]
        if periods is None:
            periods = grid.index.tolist()
        record_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("psa_ims").filter(_.record_id.isin(record_ids))
        cols = {str(p): t.pSA[int(grid.loc[p]) - 1] for p in periods}
        return (
            t.select(record_id=t.record_id, **cols).to_pandas().set_index("record_id")
        )

    def get_fas(
        self, frequencies: list[float] | None = None, **filters: Any
    ) -> pd.DataFrame:
        """Return Fourier amplitude spectra.

        Parameters
        ----------
        frequencies : list of float, optional
            Frequencies to return, in Hz. Defaults to every frequency on the grid.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_id`, one column per requested frequency.
        """
        grid = (
            self.con.table("frequencies")
            .to_pandas()
            .set_index("frequency")["freq_index"]
        )
        if frequencies is None:
            frequencies = grid.index.tolist()
        record_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("fas_ims").filter(_.record_id.isin(record_ids))
        cols = {str(f): t.FAS[int(grid.loc[f]) - 1] for f in frequencies}
        return (
            t.select(record_id=t.record_id, **cols).to_pandas().set_index("record_id")
        )

    def get_scalars(self, ims: list[str] | None = None, **filters: Any) -> pd.DataFrame:
        """Return scalar intensity measures.

        Parameters
        ----------
        ims : list of str, optional
            Which scalar IMs to return. Defaults to all of `schema.SCALAR_IMS`.
        **filters
            Passed to `get_records` to select which records to return.

        Returns
        -------
        pd.DataFrame
            Indexed by `record_id`, one column per requested IM.
        """
        ims = ims or list(schema.SCALAR_IMS)
        record_ids = self.get_records(**filters).index.tolist()
        t = self.con.table("scalars_ims").filter(_.record_id.isin(record_ids))
        return t.select("record_id", *ims).to_pandas().set_index("record_id")
