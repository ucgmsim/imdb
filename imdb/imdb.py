"""
Read and write intensity measure databases.

An intensity measure database (IMDB) is a DuckDB file holding intensity
measures (IMs) from physics-based ground-motion simulation, laid out according
to :mod:`imdb.schema`. One file holds one simulation run set.

Reading:

>>> with IMDB("path/to/ims.duckdb") as db:
...     df = db.get_im_df(["PGA", "pSA_1.0"], events=["ev1"], component="rotd50")

Writing:

>>> with IMDB.create("new.duckdb", periods=[0.1, 1.0], frequencies=[1.0]) as db:
...     db.add_events(event_df)
...     db.add_realisations(rel_df)
...     db.add_sites(site_df)
...     db.add_site_event(site_event_df)
...     db.add_records(record_df)
...     db.finalise()
"""

import contextlib
import getpass
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb
import numpy as np
import pandas as pd

from imdb import schema

logger = logging.getLogger(__name__)

_CACHED = (
    "db_meta",
    "notes",
    "im_units",
    "components",
    "periods",
    "frequencies",
    "event_ids",
    "rel_ids",
    "site_ids",
    "rel_to_event",
)

_DIM_JOINS = {
    "e": "JOIN events e ON e.event_int_id = r.event_int_id",
    "rl": "JOIN realisations rl ON rl.rel_int_id = r.rel_int_id",
    "s": "JOIN sites s ON s.site_int_id = r.site_int_id",
    "se": (
        "JOIN site_event se ON se.site_int_id = r.site_int_id "
        "AND se.event_int_id = r.event_int_id"
    ),
}

_MATCH_RTOL = 1e-6
"""Relative tolerance when matching a requested period or frequency to the grid."""


def _imdb_version() -> str:
    """Return the installed version of this library, or ``unknown``.

    Returns
    -------
    str
        Version string.
    """
    try:
        return version("imdb")
    except PackageNotFoundError:
        return "unknown"


def _parse_number(text: str) -> float:
    """Parse a period or frequency written with either ``.`` or ``p`` as the point.

    Parameters
    ----------
    text : str
        Number to parse, for example ``1.0`` or ``0p1``.

    Returns
    -------
    float
        The parsed value.
    """
    try:
        return float(text)
    except ValueError:
        return float(text.replace("p", ".", 1))


class IMDB(contextlib.AbstractContextManager):
    """An intensity measure database.

    Parameters
    ----------
    db_path : Path or str
        Path to the DuckDB file.
    read_only : bool, optional
        Open read-only. Write methods raise when true.
    memory_limit : str, optional
        DuckDB memory limit applied on open.
    """

    def __init__(
        self,
        db_path: Path | str,
        read_only: bool = True,
        memory_limit: str = "8GB",
    ) -> None:
        """Create a handle. The connection is opened lazily, or by :meth:`open`."""
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.memory_limit = memory_limit
        self._conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> Self:
        """Open the database connection.

        Returns
        -------
        IMDB
            This database, for chaining.
        """
        if self._conn is not None:
            return self
        if self.read_only and not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
        self._conn.execute("SET enable_progress_bar = false")
        return self

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._invalidate()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """The open DuckDB connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            The connection.
        """
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    def sql(
        self, query: str, params: Sequence | None = None
    ) -> duckdb.DuckDBPyRelation:
        """Run an arbitrary query against the database.

        The escape hatch for reads the keyword filters cannot express.

        Parameters
        ----------
        query : str
            SQL to execute.
        params : sequence, optional
            Prepared-statement parameters.

        Returns
        -------
        duckdb.DuckDBPyRelation
            The query result.
        """
        return self.conn.sql(query, params=params)

    def __enter__(self) -> Self:
        """Open the connection and return this database."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the connection.

        Statements autocommit, so a failed write is not rolled back here. Repair
        a partial ingest with :meth:`delete_event`, which is what makes a
        re-ingest idempotent.
        """
        self.close()

    def _invalidate(self) -> None:
        """Drop every cached lookup, after a write or a close."""
        for name in _CACHED:
            self.__dict__.pop(name, None)

    def _require_write(self) -> None:
        """Raise if the database was opened read-only."""
        if self.read_only:
            raise PermissionError(f"{self.db_path} is open read-only")

    @contextlib.contextmanager
    def _temp_frames(self, frames: Mapping[str, pd.DataFrame]):
        """Register DataFrames as views for the duration of a query.

        Parameters
        ----------
        frames : mapping of str to pandas.DataFrame
            View name to DataFrame.

        Yields
        ------
        None
        """
        for name, df in frames.items():
            self.conn.register(name, df)
        try:
            yield
        finally:
            for name in frames:
                self.conn.unregister(name)

    @contextlib.contextmanager
    def _transaction(self):
        """Group several statements into one atomic write.

        Yields
        ------
        None
        """
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # cached vocabulary
    # ------------------------------------------------------------------

    @cached_property
    def db_meta(self) -> dict[str, str]:
        """Contents of the ``db_meta`` table.

        Returns
        -------
        dict of str to str
            Key to value.
        """
        return dict(self.conn.execute("SELECT key, value FROM db_meta").fetchall())

    @cached_property
    def notes(self) -> dict[str, str]:
        """Contents of the ``notes`` table.

        Returns
        -------
        dict of str to str
            Topic to note.
        """
        return dict(self.conn.execute("SELECT topic, note FROM notes").fetchall())

    @cached_property
    def im_units(self) -> dict[str, str]:
        """Contents of the ``im_units`` table.

        Returns
        -------
        dict of str to str
            IM name to unit.
        """
        return dict(self.conn.execute("SELECT im, unit FROM im_units").fetchall())

    @cached_property
    def components(self) -> list[str]:
        """Components held by this database, from ``db_meta``.

        Returns
        -------
        list of str
            Component names.
        """
        value = self.db_meta.get("components", "")
        return [c for c in (part.strip() for part in value.split(",")) if c]

    @cached_property
    def periods(self) -> pd.Series:
        """The pSA period grid.

        Returns
        -------
        pandas.Series
            Period in seconds, indexed by 1-based ``period_index``.
        """
        return (
            self.conn.execute("SELECT period_index, period FROM periods ORDER BY 1")
            .df()
            .set_index("period_index")["period"]
        )

    @cached_property
    def frequencies(self) -> pd.Series:
        """The FAS frequency grid.

        Returns
        -------
        pandas.Series
            Frequency in Hz, indexed by 1-based ``freq_index``.
        """
        return (
            self.conn.execute(
                "SELECT freq_index, frequency FROM frequencies ORDER BY 1"
            )
            .df()
            .set_index("freq_index")["frequency"]
        )

    @cached_property
    def event_ids(self) -> pd.Series:
        """Mapping of ``event_id`` to ``event_int_id``.

        Returns
        -------
        pandas.Series
            Integer id indexed by string id.
        """
        return self._id_map("events", "event_id", "event_int_id")

    @cached_property
    def rel_ids(self) -> pd.Series:
        """Mapping of ``rel_id`` to ``rel_int_id``.

        Returns
        -------
        pandas.Series
            Integer id indexed by string id.
        """
        return self._id_map("realisations", "rel_id", "rel_int_id")

    @cached_property
    def site_ids(self) -> pd.Series:
        """Mapping of ``site_id`` to ``site_int_id``.

        Returns
        -------
        pandas.Series
            Integer id indexed by string id.
        """
        return self._id_map("sites", "site_id", "site_int_id")

    @cached_property
    def rel_to_event(self) -> pd.Series:
        """Mapping of ``rel_int_id`` to ``event_int_id``.

        Returns
        -------
        pandas.Series
            Event integer id indexed by realisation integer id.
        """
        return self._id_map("realisations", "rel_int_id", "event_int_id")

    def _id_map(self, table: str, key: str, value: str) -> pd.Series:
        """Read a two-column mapping out of a dimension table.

        Parameters
        ----------
        table : str
            Table to read.
        key : str
            Column to index by.
        value : str
            Column to map to.

        Returns
        -------
        pandas.Series
            ``value`` indexed by ``key``.
        """
        return (
            self.conn.execute(f"SELECT {key}, {value} FROM {table}")
            .df()
            .set_index(key)[value]
        )

    def _resolve(self, ids: Iterable[str], mapping: pd.Series, what: str) -> np.ndarray:
        """Resolve string ids to integer surrogates.

        Parameters
        ----------
        ids : iterable of str
            String ids to resolve.
        mapping : pandas.Series
            Mapping to resolve against.
        what : str
            Name used in the error message.

        Returns
        -------
        numpy.ndarray
            Integer ids, in the order given.
        """
        # the id columns are VARCHAR, so a numeric id from the source reaches the
        # database as its string form and must be looked up that way
        ids = np.asarray([str(value) for value in ids], dtype=object)
        unknown = pd.unique(ids[~pd.Index(ids).isin(mapping.index)])
        if len(unknown):
            raise KeyError(f"unknown {what}: {sorted(unknown)[:10]}")
        return mapping.loc[ids].to_numpy(dtype=np.int64)

    def _scalar(self, query: str, params: Sequence | None = None) -> Any:
        """Run a query returning exactly one row and one column.

        Parameters
        ----------
        query : str
            SQL to execute.
        params : sequence, optional
            Prepared-statement parameters.

        Returns
        -------
        Any
            The single value.
        """
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            raise RuntimeError(f"query returned no rows: {query}")
        return row[0]

    def _table_columns(self, table: str) -> list[str]:
        """Column names of a table, in declaration order.

        Parameters
        ----------
        table : str
            Table name.

        Returns
        -------
        list of str
            Column names.
        """
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if not rows:
            raise KeyError(f"no such table: {table}")
        return [row[0] for row in rows]

    def _grid_indices(
        self, values: Iterable[float] | None, grid: pd.Series, what: str
    ) -> tuple[list[int], list[float]]:
        """Resolve requested grid values to 1-based array indices.

        Parameters
        ----------
        values : iterable of float, optional
            Values to resolve. ``None`` returns the whole grid.
        grid : pandas.Series
            Grid to resolve against, indexed by array index.
        what : str
            Name used in the error message.

        Returns
        -------
        tuple of (list of int, list of float)
            Array indices and the matched grid values, in the order requested.
        """
        if grid.empty:
            raise KeyError(f"this database has no {what} grid")
        if values is None:
            return list(grid.index.astype(int)), list(grid.to_numpy(dtype=float))

        available = grid.to_numpy(dtype=float)
        indices, matched = [], []
        for value in values:
            value = float(value)
            close = np.flatnonzero(
                np.isclose(available, value, rtol=_MATCH_RTOL, atol=0.0)
            )
            if close.size == 0:
                raise KeyError(
                    f"{what} {value} is not on this database's grid; "
                    f"available: {np.array2string(available, threshold=20)}"
                )
            indices.append(int(grid.index[close[0]]))
            matched.append(float(available[close[0]]))
        return indices, matched

    # ------------------------------------------------------------------
    # record filtering
    # ------------------------------------------------------------------

    def _record_filter(
        self,
        need: Iterable[str] = (),
        events: Iterable[str] | None = None,
        rels: Iterable[str] | None = None,
        sites: Iterable[str] | None = None,
        component: str | Iterable[str] | None = None,
        max_rrup: float | None = None,
        record_ids: Iterable[int] | None = None,
    ) -> tuple[str, str, dict[str, pd.DataFrame]]:
        """Build the FROM and WHERE clauses shared by every record query.

        Parameters
        ----------
        need : iterable of str, optional
            Dimension aliases the caller's SELECT needs, from ``e``, ``rl``, ``s``, ``se``.
        events : iterable of str, optional
            Keep only these ``event_id`` values.
        rels : iterable of str, optional
            Keep only these ``rel_id`` values.
        sites : iterable of str, optional
            Keep only these ``site_id`` values.
        component : str or iterable of str, optional
            Keep only these components.
        max_rrup : float, optional
            Keep only records whose ``site_event.rrup`` is at most this.
        record_ids : iterable of int, optional
            Keep only these ``record_id`` values.

        Returns
        -------
        tuple of (str, str, dict of str to pandas.DataFrame)
            FROM clause, WHERE clause and the frames to register while querying.
        """
        dims, joins, wheres, frames = set(need), [], [], {}

        for alias, column, table, values, mapping in (
            ("e", "event_id", "events", events, self.event_ids),
            ("rl", "rel_id", "realisations", rels, self.rel_ids),
            ("s", "site_id", "sites", sites, self.site_ids),
        ):
            if values is None:
                continue
            dims.add(alias)
            values = [str(value) for value in values]
            self._resolve(values, mapping, column)
            view = f"_f_{table}"
            frames[view] = pd.DataFrame({"_key": np.asarray(values, dtype=object)})
            joins.append(f"JOIN {view} ON {view}._key = {alias}.{column}")

        if record_ids is not None:
            frames["_f_records"] = pd.DataFrame(
                {"_key": np.asarray(list(record_ids), dtype=np.int64)}
            )
            joins.append("JOIN _f_records ON _f_records._key = r.record_id")

        if max_rrup is not None:
            dims.add("se")
            wheres.append(f"se.rrup <= {float(max_rrup)}")

        if component is not None:
            wanted = [component] if isinstance(component, str) else list(component)
            unknown = set(wanted) - set(self.components)
            if unknown:
                raise ValueError(
                    f"components {sorted(unknown)} are not in this database; "
                    f"it holds {self.components}"
                )
            wheres.append(
                "r.component IN (" + ", ".join(f"'{c}'" for c in wanted) + ")"
            )

        from_sql = "\n".join(
            ["FROM records r"]
            + [_DIM_JOINS[alias] for alias in ("e", "rl", "s", "se") if alias in dims]
            + joins
        )
        where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
        return from_sql, where_sql, frames

    def _record_query(
        self,
        select: str,
        need: Iterable[str] = (),
        joins: Iterable[str] = (),
        **filters,
    ) -> pd.DataFrame:
        """Run a query over ``records``, indexed by ``record_id``.

        Parameters
        ----------
        select : str
            SELECT list, without the leading ``record_id``.
        need : iterable of str, optional
            Dimension aliases the SELECT needs.
        joins : iterable of str, optional
            Extra join clauses, for example onto an IM table.
        **filters
            Passed to :meth:`_record_filter`.

        Returns
        -------
        pandas.DataFrame
            Query result indexed by ``record_id``.
        """
        from_sql, where_sql, frames = self._record_filter(need=need, **filters)
        from_sql = "\n".join([from_sql, *joins])
        query = f"SELECT r.record_id, {select}\n{from_sql}\n{where_sql}"
        logger.debug("query: %s", query)
        with self._temp_frames(frames):
            return self.conn.execute(query).df().set_index("record_id")

    # ------------------------------------------------------------------
    # dimension reads
    # ------------------------------------------------------------------

    def get_events(self, expand_metadata: bool = False) -> pd.DataFrame:
        """Read the ``events`` table.

        Parameters
        ----------
        expand_metadata : bool, optional
            Expand the JSON ``metadata`` column into columns.

        Returns
        -------
        pandas.DataFrame
            Events indexed by ``event_int_id``.
        """
        df = self.conn.execute("SELECT * FROM events").df().set_index("event_int_id")
        return _expand_metadata(df) if expand_metadata else df

    def get_realisations(self, expand_metadata: bool = False) -> pd.DataFrame:
        """Read the ``realisations`` table.

        Parameters
        ----------
        expand_metadata : bool, optional
            Expand the JSON ``metadata`` column into columns.

        Returns
        -------
        pandas.DataFrame
            Realisations indexed by ``rel_int_id``.
        """
        df = (
            self.conn.execute("SELECT * FROM realisations").df().set_index("rel_int_id")
        )
        return _expand_metadata(df) if expand_metadata else df

    def get_sites(self, expand_metadata: bool = False) -> pd.DataFrame:
        """Read the ``sites`` table.

        Parameters
        ----------
        expand_metadata : bool, optional
            Expand the JSON ``metadata`` column into columns.

        Returns
        -------
        pandas.DataFrame
            Sites indexed by ``site_int_id``.
        """
        df = self.conn.execute("SELECT * FROM sites").df().set_index("site_int_id")
        return _expand_metadata(df) if expand_metadata else df

    def get_site_event(
        self,
        sites: Iterable[str] | None = None,
        events: Iterable[str] | None = None,
        max_rrup: float | None = None,
        expand_metadata: bool = False,
    ) -> pd.DataFrame:
        """Read the ``site_event`` table.

        Parameters
        ----------
        sites : iterable of str, optional
            Keep only these ``site_id`` values.
        events : iterable of str, optional
            Keep only these ``event_id`` values.
        max_rrup : float, optional
            Keep only pairs whose ``rrup`` is at most this.
        expand_metadata : bool, optional
            Expand the JSON ``metadata`` column into columns.

        Returns
        -------
        pandas.DataFrame
            Site-event pairs, with ``site_id`` and ``event_id`` added.
        """
        wheres, frames = [], {}
        if sites is not None:
            sites = [str(value) for value in sites]
            self._resolve(sites, self.site_ids, "site_id")
            frames["_f_sites"] = pd.DataFrame({"_key": np.asarray(sites, dtype=object)})
            wheres.append("s.site_id IN (SELECT _key FROM _f_sites)")
        if events is not None:
            events = [str(value) for value in events]
            self._resolve(events, self.event_ids, "event_id")
            frames["_f_events"] = pd.DataFrame(
                {"_key": np.asarray(events, dtype=object)}
            )
            wheres.append("e.event_id IN (SELECT _key FROM _f_events)")
        if max_rrup is not None:
            wheres.append(f"se.rrup <= {float(max_rrup)}")
        where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""

        query = f"""
            SELECT se.*, s.site_id, e.event_id
            FROM site_event se
            JOIN sites s  ON s.site_int_id  = se.site_int_id
            JOIN events e ON e.event_int_id = se.event_int_id
            {where_sql}
        """
        with self._temp_frames(frames):
            df = self.conn.execute(query).df()
        return _expand_metadata(df) if expand_metadata else df

    # ------------------------------------------------------------------
    # record and IM reads
    # ------------------------------------------------------------------

    def get_records(self, **filters) -> pd.DataFrame:
        """Read record identities.

        Parameters
        ----------
        **filters
            Any of ``events``, ``rels``, ``sites``, ``component``, ``max_rrup``,
            ``record_ids``.

        Returns
        -------
        pandas.DataFrame
            ``event_id``, ``rel_id``, ``site_id`` and ``component``, indexed by
            ``record_id``.
        """
        return self._record_query(
            "e.event_id, rl.rel_id, s.site_id, r.component",
            need=("e", "rl", "s"),
            **filters,
        )

    def get_psa(
        self, periods: Iterable[float] | None = None, **filters
    ) -> pd.DataFrame:
        """Read pSA values.

        Parameters
        ----------
        periods : iterable of float, optional
            Periods in seconds. ``None`` reads the whole grid.
        **filters
            Any of ``events``, ``rels``, ``sites``, ``component``, ``max_rrup``,
            ``record_ids``.

        Returns
        -------
        pandas.DataFrame
            pSA in g, indexed by ``record_id``, one column per period, labelled
            with the period in seconds. Records with no ``psa_ims`` row are absent.
        """
        return self._spectral_read("psa_ims", "pSA", self.periods, periods, **filters)

    def get_fas(
        self, frequencies: Iterable[float] | None = None, **filters
    ) -> pd.DataFrame:
        """Read FAS values.

        Parameters
        ----------
        frequencies : iterable of float, optional
            Frequencies in Hz. ``None`` reads the whole grid.
        **filters
            Any of ``events``, ``rels``, ``sites``, ``component``, ``max_rrup``,
            ``record_ids``.

        Returns
        -------
        pandas.DataFrame
            FAS in g.s, indexed by ``record_id``, one column per frequency,
            labelled with the frequency in Hz. Records with no ``fas_ims`` row
            are absent.
        """
        return self._spectral_read(
            "fas_ims", "FAS", self.frequencies, frequencies, **filters
        )

    def _spectral_read(
        self,
        table: str,
        column: str,
        grid: pd.Series,
        values: Iterable[float] | None,
        **filters,
    ) -> pd.DataFrame:
        """Project selected array elements out of a spectral IM table.

        Parameters
        ----------
        table : str
            IM table to read.
        column : str
            Array column in that table.
        grid : pandas.Series
            The database's grid for that column.
        values : iterable of float, optional
            Grid values to read. ``None`` reads all of them.
        **filters
            Passed to :meth:`_record_filter`.

        Returns
        -------
        pandas.DataFrame
            One column per requested grid value, indexed by ``record_id``.
        """
        indices, matched = self._grid_indices(values, grid, column)
        select = ", ".join(f'im.{column}[{i}] AS "c{n}"' for n, i in enumerate(indices))
        df = self._record_query(
            select, joins=[f"JOIN {table} im USING (record_id)"], **filters
        )
        df.columns = pd.Index(matched, name=column)
        return df

    def get_scalars(self, ims: Iterable[str] | None = None, **filters) -> pd.DataFrame:
        """Read scalar IM values.

        Parameters
        ----------
        ims : iterable of str, optional
            Scalar IM names. ``None`` reads all of them.
        **filters
            Any of ``events``, ``rels``, ``sites``, ``component``, ``max_rrup``,
            ``record_ids``.

        Returns
        -------
        pandas.DataFrame
            One column per requested IM, indexed by ``record_id``. Records with
            no ``scalars_ims`` row are absent. ``CAV``, ``AI``, ``Ds575`` and
            ``Ds595`` are NULL for ``rotd*`` components.
        """
        wanted = list(schema.SCALAR_IMS) if ims is None else list(ims)
        unknown = set(wanted) - set(schema.SCALAR_IMS)
        if unknown:
            raise ValueError(
                f"unknown scalar IMs {sorted(unknown)}; known: {list(schema.SCALAR_IMS)}"
            )
        select = ", ".join(f"im.{im}" for im in wanted)
        return self._record_query(
            select, joins=["JOIN scalars_ims im USING (record_id)"], **filters
        )

    def get_im_df(self, ims: Iterable[str], **filters) -> pd.DataFrame:
        """Read named intensity measures into one DataFrame.

        Names are ``PGA``, ``PGV``, ``PGD``, ``CAV``, ``AI``, ``Ds575``,
        ``Ds595`` for scalars, ``pSA_<period>`` for response spectra and
        ``FAS_<frequency>`` for Fourier spectra. The point may be written as
        ``.`` or ``p``, so ``pSA_0.1`` and ``pSA_0p1`` are the same.

        Parameters
        ----------
        ims : iterable of str
            IM names to read.
        **filters
            Any of ``events``, ``rels``, ``sites``, ``component``, ``max_rrup``,
            ``record_ids``.

        Returns
        -------
        pandas.DataFrame
            One column per requested name, in the order requested, indexed by
            ``record_id``. IM tables are joined outer, so a record missing from
            one of them reads as NaN in its columns.
        """
        ims = list(ims)
        scalars, psa, fas = [], [], []
        for name in ims:
            if name in schema.SCALAR_IMS:
                scalars.append(name)
            elif name.startswith("pSA_"):
                psa.append((name, _parse_number(name[4:])))
            elif name.startswith("FAS_"):
                fas.append((name, _parse_number(name[4:])))
            else:
                raise ValueError(
                    f"cannot parse IM name {name!r}; expected one of "
                    f"{list(schema.SCALAR_IMS)}, pSA_<period> or FAS_<frequency>"
                )

        parts = []
        if scalars:
            parts.append(self.get_scalars(ims=scalars, **filters))
        for getter, requested in ((self.get_psa, psa), (self.get_fas, fas)):
            if not requested:
                continue
            part = getter([value for _, value in requested], **filters)
            part.columns = pd.Index([name for name, _ in requested])
            parts.append(part)

        df = parts[0] if len(parts) == 1 else pd.concat(parts, axis=1, join="outer")
        return df[ims]

    # ------------------------------------------------------------------
    # creation
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        db_path: Path | str,
        periods: Iterable[float],
        frequencies: Iterable[float] = (),
        components: Iterable[str] = schema.COMPONENTS,
        db_meta: Mapping[str, object] | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> Self:
        """Create an empty database and return it open for writing.

        Parameters
        ----------
        db_path : Path or str
            Path of the file to create.
        periods : iterable of float
            pSA period grid in seconds. Sorted ascending on write.
        frequencies : iterable of float, optional
            FAS frequency grid in Hz. Sorted ascending on write.
        components : iterable of str, optional
            Components this database will hold.
        db_meta : mapping, optional
            Values merged over the defaults, for example ``dataset_description``,
            ``source`` and the ``*_metadata_keys`` declarations.
        overwrite : bool, optional
            Replace an existing file.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        IMDB
            The new database, open for writing.
        """
        db_path = Path(db_path)
        if db_path.exists():
            if not overwrite:
                raise FileExistsError(db_path)
            db_path.unlink()

        unknown = set(components) - set(schema.COMPONENTS)
        if unknown:
            raise ValueError(
                f"unknown components {sorted(unknown)}; known: {list(schema.COMPONENTS)}"
            )

        db = cls(db_path, read_only=False, **kwargs).open()
        db.conn.execute(schema.DDL)

        periods = np.unique(np.asarray(list(periods), dtype=float))
        frequencies = np.unique(np.asarray(list(frequencies), dtype=float))
        db._insert(
            "periods",
            pd.DataFrame(
                {"period_index": np.arange(1, len(periods) + 1), "period": periods}
            ),
        )
        db._insert(
            "frequencies",
            pd.DataFrame(
                {
                    "freq_index": np.arange(1, len(frequencies) + 1),
                    "frequency": frequencies,
                }
            ),
        )
        db._insert(
            "im_units",
            pd.DataFrame(
                {"im": list(schema.IM_UNITS), "unit": list(schema.IM_UNITS.values())}
            ),
        )
        db._insert(
            "notes",
            pd.DataFrame(
                {"topic": list(schema.NOTES), "note": list(schema.NOTES.values())}
            ),
        )

        meta: dict[str, str] = {
            "schema_version": schema.SCHEMA_VERSION,
            "dataset_id": db_path.stem,
            "dataset_description": "",
            "components": ",".join(components),
            "n_periods": str(len(periods)),
            "n_frequencies": str(len(frequencies)),
            "sort_order": "event",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "creator": getpass.getuser(),
            "source": "",
            "imdb_version": _imdb_version(),
            **{key: "" for key in schema.METADATA_TABLES.values()},
        }
        meta.update({key: str(value) for key, value in (db_meta or {}).items()})
        db.set_db_meta(meta)
        return db

    def set_db_meta(self, values: Mapping[str, object]) -> None:
        """Insert or replace ``db_meta`` entries.

        Parameters
        ----------
        values : mapping
            Keys and values to write. Values are stored as strings.
        """
        self._require_write()
        frame = pd.DataFrame(
            {
                "key": list(values),
                "value": [str(value) for value in values.values()],
            }
        )
        with self._temp_frames({"_meta": frame}):
            self.conn.execute(
                "INSERT OR REPLACE INTO db_meta SELECT key, value FROM _meta"
            )
        self._invalidate()

    def _insert(self, table: str, df: pd.DataFrame) -> None:
        """Insert a DataFrame whose columns match the table's, in order.

        Parameters
        ----------
        table : str
            Target table.
        df : pandas.DataFrame
            Rows to insert.
        """
        if df.empty:
            return
        with self._temp_frames({"_rows": df}):
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM _rows")

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def _prepare(
        self, df: pd.DataFrame, table: str, required: Sequence[str]
    ) -> pd.DataFrame:
        """Validate and order a DataFrame against a table's columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Rows to write.
        table : str
            Target table.
        required : sequence of str
            Columns that must be present.

        Returns
        -------
        pandas.DataFrame
            The rows, with missing columns added as NULL and ordered to match
            the table.
        """
        columns = self._table_columns(table)
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"{table} rows are missing columns {missing}")
        unknown = [name for name in df.columns if name not in columns]
        if unknown:
            raise ValueError(
                f"columns {unknown} are not in {table}; extra fields belong in metadata"
            )
        df = df.copy()
        if "metadata" in columns:
            df["metadata"] = self._pack_metadata(df.get("metadata"), table)
        for name in columns:
            if name not in df.columns:
                df[name] = None
        return df[columns]

    def _pack_metadata(self, values: pd.Series | None, table: str) -> pd.Series | None:
        """Serialise a metadata column to JSON and reconcile its declared keys.

        The first write of a table's metadata declares the permitted keys in
        ``db_meta``; later writes are validated against that declaration.

        Parameters
        ----------
        values : pandas.Series or None
            Metadata as dicts or JSON strings.
        table : str
            Table being written, used to find the ``db_meta`` key.

        Returns
        -------
        pandas.Series or None
            JSON strings, or ``None`` if there was nothing to pack.
        """
        if values is None:
            return None
        packed, seen = [], set()
        for value in values:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                packed.append(None)
                continue
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise TypeError(
                    f"{table}.metadata must hold dicts or JSON objects, got {type(value)}"
                )
            seen.update(value)
            packed.append(json.dumps(value, sort_keys=True, default=str))

        meta_key = schema.METADATA_TABLES[table]
        declared = {k for k in self.db_meta.get(meta_key, "").split(",") if k}
        if declared:
            undeclared = seen - declared
            if undeclared:
                raise ValueError(
                    f"{table}.metadata keys {sorted(undeclared)} are not declared in "
                    f"db_meta.{meta_key} ({sorted(declared)})"
                )
        elif seen:
            self.set_db_meta({meta_key: ",".join(sorted(seen))})
        return pd.Series(packed, index=values.index, dtype=object)

    def _next_int_ids(self, table: str, column: str, count: int) -> np.ndarray:
        """Allocate contiguous integer surrogates for a dimension table.

        Parameters
        ----------
        table : str
            Table to extend.
        column : str
            Surrogate key column.
        count : int
            How many ids to allocate.

        Returns
        -------
        numpy.ndarray
            The new ids.
        """
        current = self._scalar(f"SELECT max({column}) FROM {table}")
        start = 1 if current is None else int(current) + 1
        return np.arange(start, start + count, dtype=np.int64)

    def add_events(self, df: pd.DataFrame) -> None:
        """Insert rows into ``events``.

        Parameters
        ----------
        df : pandas.DataFrame
            Requires ``event_id``. Other columns must be ``events`` columns;
            extra fields go in a ``metadata`` dict column. ``event_int_id`` is
            assigned here and must not be supplied.
        """
        self._require_write()
        df = self._prepare(df, "events", ["event_id"]).drop(columns="event_int_id")
        df["event_id"] = df["event_id"].astype(str)
        self._reject_existing(df["event_id"], self.event_ids, "event_id")
        df.insert(
            0, "event_int_id", self._next_int_ids("events", "event_int_id", len(df))
        )
        self._insert("events", df)
        self._invalidate()
        logger.info("inserted %d events", len(df))

    def add_realisations(self, df: pd.DataFrame) -> None:
        """Insert rows into ``realisations``.

        Parameters
        ----------
        df : pandas.DataFrame
            Requires ``rel_id`` and ``event_id``. ``rel_int_id`` is assigned
            here and must not be supplied.
        """
        self._require_write()
        df = df.copy()
        if "event_id" not in df.columns:
            raise ValueError("realisation rows are missing column ['event_id']")
        event_int_id = self._resolve(df.pop("event_id"), self.event_ids, "event_id")
        df["event_int_id"] = event_int_id
        df = self._prepare(df, "realisations", ["rel_id"]).drop(columns="rel_int_id")
        df["rel_id"] = df["rel_id"].astype(str)
        self._reject_existing(df["rel_id"], self.rel_ids, "rel_id")
        df.insert(
            0, "rel_int_id", self._next_int_ids("realisations", "rel_int_id", len(df))
        )
        self._insert("realisations", df)
        self._invalidate()
        logger.info("inserted %d realisations", len(df))

    def add_sites(self, df: pd.DataFrame) -> None:
        """Insert rows into ``sites``.

        Parameters
        ----------
        df : pandas.DataFrame
            Requires ``site_id``, ``lat`` and ``lon``. ``site_int_id`` is
            assigned here and must not be supplied.
        """
        self._require_write()
        df = self._prepare(df, "sites", ["site_id", "lat", "lon"]).drop(
            columns="site_int_id"
        )
        df["site_id"] = df["site_id"].astype(str)
        self._reject_existing(df["site_id"], self.site_ids, "site_id")
        df.insert(0, "site_int_id", self._next_int_ids("sites", "site_int_id", len(df)))
        self._insert("sites", df)
        self._invalidate()
        logger.info("inserted %d sites", len(df))

    def add_site_event(self, df: pd.DataFrame) -> None:
        """Insert rows into ``site_event``.

        Parameters
        ----------
        df : pandas.DataFrame
            Requires ``site_id`` and ``event_id``, which are resolved to their
            integer surrogates here.
        """
        self._require_write()
        df = df.copy()
        for column in ("site_id", "event_id"):
            if column not in df.columns:
                raise ValueError(f"site_event rows are missing column ['{column}']")
        df["site_int_id"] = self._resolve(df.pop("site_id"), self.site_ids, "site_id")
        df["event_int_id"] = self._resolve(
            df.pop("event_id"), self.event_ids, "event_id"
        )
        df = self._prepare(df, "site_event", ["site_int_id", "event_int_id"])
        self._insert("site_event", df)
        logger.info("inserted %d site-event pairs", len(df))

    def add_records(self, df: pd.DataFrame) -> np.ndarray:
        """Insert records and their IM values.

        Writes ``records`` plus whichever of ``psa_ims``, ``fas_ims`` and
        ``scalars_ims`` the input covers. ``CAV``, ``AI``, ``Ds575`` and
        ``Ds595`` are set to NULL on ``rotd*`` rows.

        Parameters
        ----------
        df : pandas.DataFrame
            Requires ``rel_id``, ``site_id`` and ``component``. May carry a
            ``pSA`` column of arrays, a ``FAS`` column of arrays, and any of the
            seven scalar IM columns.

        Returns
        -------
        numpy.ndarray
            The ``record_id`` assigned to each row, in input order.
        """
        self._require_write()
        df = df.copy()
        for column in ("rel_id", "site_id", "component"):
            if column not in df.columns:
                raise ValueError(f"record rows are missing column ['{column}']")

        component = df["component"].astype(str)
        unknown = set(component.unique()) - set(self.components)
        if unknown:
            raise ValueError(
                f"components {sorted(unknown)} are not declared in db_meta.components "
                f"({self.components})"
            )

        rel_int_id = self._resolve(df["rel_id"], self.rel_ids, "rel_id")
        site_int_id = self._resolve(df["site_id"], self.site_ids, "site_id")
        event_int_id = self.rel_to_event.loc[rel_int_id].to_numpy(dtype=np.int64)

        known = {"rel_id", "site_id", "component", "pSA", "FAS", *schema.SCALAR_IMS}
        unknown_columns = [name for name in df.columns if name not in known]
        if unknown_columns:
            raise ValueError(
                f"columns {unknown_columns} are not record or IM columns; expected "
                f"rel_id, site_id, component, pSA, FAS or one of {list(schema.SCALAR_IMS)}"
            )

        record_id = (
            self.conn.execute(
                "SELECT nextval('record_id_seq') AS record_id FROM range(?)", [len(df)]
            )
            .df()["record_id"]
            .to_numpy(dtype=np.int64)
        )

        with self._transaction():
            self._write_record_tables(
                df, record_id, event_int_id, rel_int_id, site_int_id, component
            )

        logger.info("inserted %d records", len(df))
        return record_id

    def _write_record_tables(
        self,
        df: pd.DataFrame,
        record_id: np.ndarray,
        event_int_id: np.ndarray,
        rel_int_id: np.ndarray,
        site_int_id: np.ndarray,
        component: pd.Series,
    ) -> None:
        """Insert one batch into ``records`` and the IM tables it covers.

        Parameters
        ----------
        df : pandas.DataFrame
            The validated input rows.
        record_id : numpy.ndarray
            Allocated record ids.
        event_int_id : numpy.ndarray
            Event surrogate of each row.
        rel_int_id : numpy.ndarray
            Realisation surrogate of each row.
        site_int_id : numpy.ndarray
            Site surrogate of each row.
        component : pandas.Series
            Component of each row.
        """
        self._insert(
            "records",
            pd.DataFrame(
                {
                    "record_id": record_id,
                    "event_int_id": event_int_id,
                    "rel_int_id": rel_int_id,
                    "site_int_id": site_int_id,
                    "component": component.to_numpy(dtype=object),
                }
            ),
        )

        for column, table, grid in (
            ("pSA", "psa_ims", self.periods),
            ("FAS", "fas_ims", self.frequencies),
        ):
            if column not in df.columns:
                continue
            # a row with no array simply gets no row in this IM table, which is how
            # the schema expresses per-record IM coverage
            present = df[column].notna().to_numpy()
            if not present.any():
                continue
            arrays = [
                np.asarray(value, dtype=np.float32) for value in df.loc[present, column]
            ]
            bad = {array.size for array in arrays} - {len(grid)}
            if bad:
                raise ValueError(
                    f"{column} arrays have lengths {sorted(bad)} but this database's "
                    f"grid has {len(grid)} entries"
                )
            self._insert(
                table,
                pd.DataFrame(
                    {
                        "record_id": record_id[present],
                        column: pd.Series(arrays, dtype=object),
                    }
                ),
            )

        present = [im for im in schema.SCALAR_IMS if im in df.columns]
        if present:
            scalars = pd.DataFrame({"record_id": record_id})
            is_rotd = component.str.startswith("rotd").to_numpy()
            for im in schema.SCALAR_IMS:
                values = (
                    pd.to_numeric(df[im], errors="raise").astype("float32")
                    if im in present
                    else pd.Series(np.nan, index=df.index, dtype="float32")
                )
                values = values.to_numpy(dtype=np.float32, copy=True)
                if im in schema.ROTD_UNDEFINED:
                    values[is_rotd] = np.nan
                scalars[im] = values
            self._insert("scalars_ims", scalars)

    def _reject_existing(self, ids: pd.Series, mapping: pd.Series, what: str) -> None:
        """Raise if any of these string ids is already in the database.

        Parameters
        ----------
        ids : pandas.Series
            String ids about to be written.
        mapping : pandas.Series
            Existing ids, as the index.
        what : str
            Name used in the error message.
        """
        duplicated = ids[ids.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"duplicate {what} in the input: {sorted(duplicated)[:10]}"
            )
        existing = ids[ids.isin(mapping.index)].unique()
        if len(existing):
            raise ValueError(
                f"{what} already in the database: {sorted(existing)[:10]}; "
                "use delete_event() to re-ingest"
            )

    def delete_event(self, event_id: str) -> None:
        """Delete an event and everything derived from it.

        Removes the event's rows from ``psa_ims``, ``fas_ims``, ``scalars_ims``,
        ``records``, ``site_event``, ``realisations`` and ``events``, so a
        re-ingest is a delete followed by the same sequence of ``add_*`` calls.
        Sites are shared across events and are never deleted.

        Parameters
        ----------
        event_id : str
            The event to delete.
        """
        self._require_write()
        event_int_id = int(self._resolve([event_id], self.event_ids, "event_id")[0])
        for table in ("psa_ims", "fas_ims", "scalars_ims"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE record_id IN "
                "(SELECT record_id FROM records WHERE event_int_id = ?)",
                [event_int_id],
            )
        for table in ("records", "site_event", "realisations", "events"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE event_int_id = ?",
                [event_int_id],
            )
        self._invalidate()
        logger.info("deleted event %s", event_id)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check the invariants the large tables do not enforce as constraints.

        Works read-only. Returns problems rather than raising, so an ingest
        script can report all of them at once.

        Returns
        -------
        list of str
            One line per problem found. Empty when the database is consistent.
        """
        problems = []

        def count(query: str) -> int:
            return int(self._scalar(query))

        for column, table, key in (
            ("rel_int_id", "realisations", "rel_int_id"),
            ("site_int_id", "sites", "site_int_id"),
            ("event_int_id", "events", "event_int_id"),
        ):
            n = count(
                f"SELECT count(*) FROM records r "
                f"LEFT JOIN {table} d ON d.{key} = r.{column} WHERE d.{key} IS NULL"
            )
            if n:
                problems.append(f"records: {n} rows with an orphan {column}")

        for column, table, key in (
            ("site_int_id", "sites", "site_int_id"),
            ("event_int_id", "events", "event_int_id"),
        ):
            n = count(
                f"SELECT count(*) FROM site_event se "
                f"LEFT JOIN {table} d ON d.{key} = se.{column} WHERE d.{key} IS NULL"
            )
            if n:
                problems.append(f"site_event: {n} rows with an orphan {column}")

        n = count(
            "SELECT count(*) FROM records r JOIN realisations rl USING (rel_int_id) "
            "WHERE r.event_int_id != rl.event_int_id"
        )
        if n:
            problems.append(
                f"records: {n} rows whose event_int_id disagrees with their realisation"
            )

        for table in schema.IM_TABLES:
            n = count(
                f"SELECT count(*) FROM {table} im "
                "LEFT JOIN records r USING (record_id) WHERE r.record_id IS NULL"
            )
            if n:
                problems.append(f"{table}: {n} rows with no matching record")

        for table, column, grid in (
            ("psa_ims", "pSA", "periods"),
            ("fas_ims", "FAS", "frequencies"),
        ):
            n = count(
                f"SELECT count(*) FROM {table} "
                f"WHERE len({column}) != (SELECT count(*) FROM {grid})"
            )
            if n:
                problems.append(f"{table}: {n} rows whose {column} length is wrong")

        n = count(
            "SELECT count(*) FROM (SELECT 1 FROM records "
            "GROUP BY rel_int_id, site_int_id, component HAVING count(*) > 1)"
        )
        if n:
            problems.append(
                f"records: {n} duplicated (rel_int_id, site_int_id, component) keys"
            )

        n = count(
            "SELECT count(*) FROM (SELECT 1 FROM site_event "
            "GROUP BY site_int_id, event_int_id HAVING count(*) > 1)"
        )
        if n:
            problems.append(
                f"site_event: {n} duplicated (site_int_id, event_int_id) keys"
            )

        for table in schema.IM_TABLES:
            n = count(
                f"SELECT count(*) FROM (SELECT 1 FROM {table} "
                "GROUP BY record_id HAVING count(*) > 1)"
            )
            if n:
                problems.append(f"{table}: {n} record_ids with more than one row")

        declared = set(self.components)
        found = {
            row[0]
            for row in self.conn.execute(
                "SELECT DISTINCT component FROM records"
            ).fetchall()
        }
        if found - declared:
            problems.append(
                f"records: components {sorted(found - declared)} are not in "
                f"db_meta.components ({sorted(declared)})"
            )

        for table, meta_key in schema.METADATA_TABLES.items():
            allowed = {k for k in self.db_meta.get(meta_key, "").split(",") if k}
            keys = {
                row[0]
                for row in self.conn.execute(
                    f"SELECT DISTINCT unnest(json_keys(metadata)) FROM {table} "
                    "WHERE metadata IS NOT NULL"
                ).fetchall()
            }
            if keys - allowed:
                problems.append(
                    f"{table}.metadata: keys {sorted(keys - allowed)} are not declared "
                    f"in db_meta.{meta_key}"
                )

        for key, table in (("n_periods", "periods"), ("n_frequencies", "frequencies")):
            declared_n = self.db_meta.get(key)
            actual = count(f"SELECT count(*) FROM {table}")
            if declared_n is not None and int(declared_n) != actual:
                problems.append(
                    f"db_meta.{key} is {declared_n} but {table} has {actual} rows"
                )

        return problems

    def finalise(self) -> None:
        """Refresh the derived ``db_meta`` counts, validate, and checkpoint.

        Call once after the last write.
        """
        self._require_write()
        self.set_db_meta(
            {
                "n_periods": len(self.periods),
                "n_frequencies": len(self.frequencies),
            }
        )
        problems = self.validate()
        if problems:
            raise ValueError("database is inconsistent:\n  " + "\n  ".join(problems))
        self.conn.execute("CHECKPOINT")
        logger.info("finalised %s", self.db_path)


def _expand_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a JSON ``metadata`` column into columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with a ``metadata`` column of JSON strings.

    Returns
    -------
    pandas.DataFrame
        The frame with ``metadata`` replaced by its fields.
    """
    if "metadata" not in df.columns:
        return df
    parsed = [
        json.loads(value) if isinstance(value, str) else {} for value in df["metadata"]
    ]
    expanded = pd.json_normalize(parsed)
    expanded.index = df.index
    return pd.concat([df.drop(columns="metadata"), expanded], axis=1)
