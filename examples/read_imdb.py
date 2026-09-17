"""Read the different kinds of data an IMDB holds: dimensions, records, spectra and scalars.

Run against a real database, e.g.:

    uv run python examples/read_imdb.py /path/to/cs25p6_imdb.duckdb
"""

import argparse
from pathlib import Path

from imdb import IMDB


def main(db_path: Path) -> None:
    """Print each kind of data an IMDB holds, for a single event.

    Parameters
    ----------
    db_path : Path
        Path to an IMDB.
    """
    with IMDB(db_path) as db:
        # db_meta: dataset-level metadata (schema version, components held, when it was built).
        print("db_meta:", db.db_meta)

        # Dimensions: events, realisations, sites. Each is indexed by its stable id.
        events = db.get_events()
        print(f"\n{len(events)} events, columns: {list(events.columns)}")
        event_id = events.index[0]
        print(events.loc[event_id])

        realisations = db.get_realisations()
        print(
            f"\n{len(realisations)} realisations, columns: {list(realisations.columns)}"
        )

        sites = db.get_sites()
        print(f"\n{len(sites)} sites, columns: {list(sites.columns)}")

        # site_event: per (site, event) distance measures (rrup, rjb, rx, ry).
        site_event = db.get_site_event(event_ids=[event_id])
        print(f"\nsite_event rows for {event_id}: {len(site_event)}")

        # Records: one row per (rel_id, site_id, component, kind, gmm_key). `kind` distinguishes
        # physics-based simulation, empirical GMM prediction, and observed ground motion.
        records = db.get_records(event_ids=[event_id])
        print(f"\nrecord kinds for {event_id}:")
        print(records["kind"].value_counts())

        # Response spectra (pSA) and scalar IMs, filtered the same way as get_records: any of its
        # keyword filters (event_ids, rel_ids, site_ids, component, kind, gmm_key) can be passed
        # straight through via **filters.
        simulated_psa = db.get_psa(
            event_ids=[event_id], kind="simulated", component="geom"
        )
        print(f"\nsimulated pSA (geom component), {len(simulated_psa)} records:")
        print(simulated_psa.iloc[:3, :5])

        # GMM predictions carry a ln-space total sigma alongside each IM (`sigma=True`)
        gmm_psa = db.get_psa(
            event_ids=[event_id], sigma=True, kind="gmm", component="rotd50"
        )
        print(f"\nGMM pSA with sigma (rotd50 component), {len(gmm_psa)} records:")
        print(gmm_psa.iloc[:3, :4])

        simulated_scalars = db.get_scalars(
            ims=["PGA", "PGV"], event_ids=[event_id], kind="simulated", component="geom"
        )
        print(
            f"\nsimulated scalar IMs (geom component), {len(simulated_scalars)} records:"
        )
        print(simulated_scalars.head(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path, help="Path to an IMDB")
    args = parser.parse_args()
    main(args.db_path)
