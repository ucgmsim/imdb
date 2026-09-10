#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2",
#     "pandas>=3",
#     "tqdm",
#     "qcore-utils",
#     "source-modelling",
#     "imdb",
#     "oq-wrapper",
# ]
#
# [tool.uv.sources]
# imdb = { git = "ssh://git@github.com/ucgmsim/imdb.git", branch = "emp-gmm-support" }
# ///

"""Ingest NZ NSHM 2010 fault ruptures (source_data/im_data) into an IMDB.

Every fault with simulated IM output (a directory under `im_data/`) is ingested;
base (non-REL) Srf/IM files are skipped, only numbered realisations count. A fault
missing its source_data or NZ_FLTmodel_2010.txt entry is skipped with a warning.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import oq_wrapper as oqw
import pandas as pd
from tqdm import tqdm

from imdb import IMDB
from qcore import nhm
from source_modelling.sources import Fault

# oq_wrapper warns per-row when falling back to active-shallow GMMs for VOLCANIC tect
# type; that's expected for this dataset (NSHM2022 has no dedicated volcanic models).
warnings.filterwarnings(
    "ignore", message="Using active_shallow type model for VOLCANIC tectonic type", category=UserWarning
)

SCALAR_COLS = ["PGA", "PGV", "CAV", "AI", "Ds575", "Ds595"]


def run_pSA_logic_tree(rupture_df: pd.DataFrame, tect_type: str, periods: list[float]):
    """Run the NSHM2022 pSA GM logic tree: the weighted combination plus every individual GMM/branch.

    NSHM2022's logic tree config only defines weights for pSA (not PGA/PGV), so that's
    the only IM available through `run_gmm_logic_tree`.
    """
    weighted, ind = oqw.run_gmm_logic_tree(
        oqw.constants.GMMLogicTree.NSHM2022,
        oqw.constants.TectType[tect_type],
        rupture_df,
        "pSA",
        periods=periods,
        return_ind_results=True,
    )
    mean_cols = [f"pSA_{p}_mean" for p in periods]
    sigma_cols = [f"pSA_{p}_std_Total" for p in periods]

    gmm_rows, psa_rows, psa_sigma_rows = [], [], []
    for gmm_key, result in {"NSHM2022": weighted, **{k: df for k, (_, df) in ind.items()}}.items():
        gmm_rows.append(
            pd.DataFrame(
                {
                    "rel_id": rupture_df["rel_id"],
                    "site_id": rupture_df["site_id"],
                    "component": "rotd50",
                    "kind": "gmm",
                    "gmm_key": gmm_key,
                }
            )
        )
        psa_rows.append(np.exp(result[mean_cols].to_numpy()))
        psa_sigma_rows.append(result[sigma_cols].to_numpy())

    return pd.concat(gmm_rows, ignore_index=True), np.concatenate(psa_rows), np.concatenate(psa_sigma_rows)


def load_sites(data: Path) -> pd.DataFrame:
    """Load the station lon/lat, vs30 and z1.0/z2.5, indexed by station code."""
    stem = data / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land"
    ll = pd.read_csv(f"{stem}.ll", sep=r"\s+", header=None, names=["lon", "lat", "station"])
    vs30 = pd.read_csv(f"{stem}.vs30", sep=r"\s+", header=None, names=["station", "vs30"])
    z = pd.read_csv(f"{stem}.z").rename(
        columns={"Station_Name": "station", "Z_1.0(km)": "z1p0", "Z_2.5(km)": "z2p5"}
    )
    return ll.merge(vs30, on="station").merge(z[["station", "z1p0", "z2p5"]], on="station").set_index("station")


def rel_number(path: Path) -> int:
    """Extract the realisation number from a `<fault>_RELnn.csv` filename."""
    match = re.search(r"_REL(\d+)\.csv$", path.name)
    assert match is not None
    return int(match.group(1))


def discover_faults(data: Path) -> list[str]:
    """Faults with simulated IM output: every directory under `im_data/`."""
    return sorted(p.name for p in (data / "im_data").iterdir() if p.is_dir())


def missing_reason(data: Path, fault_name: str, nhm_faults: set[str]) -> str | None:
    """Why `fault_name` can't be ingested, or `None` if it has everything required."""
    if fault_name not in nhm_faults:
        return "not present in NZ_FLTmodel_2010.txt"
    srf_dir = data / "source_data" / fault_name / "Srf"
    if not (srf_dir / f"{fault_name}.csv").exists():
        return f"missing {srf_dir / f'{fault_name}.csv'}"
    if not any(srf_dir.glob(f"{fault_name}_REL*.csv")):
        return f"no {fault_name}_REL*.csv realisation files in {srf_dir}"
    return None


def discover_periods(data: Path, fault_name: str) -> list[float]:
    """pSA periods, read from the first realisation's IM file (same across all faults)."""
    im_dir = data / "im_data" / fault_name / "IM"
    first_rel = sorted(im_dir.glob(f"{fault_name}_REL*.csv"), key=rel_number)[0]
    columns = pd.read_csv(first_rel, nrows=0).columns
    return [float(c.removeprefix("pSA_")) for c in columns if c.startswith("pSA_")]


def main(data: Path, db_path: Path) -> None:
    sites_all = load_sites(data)
    nhm_faults = nhm.load_nhm(str(data / "NZ_FLTmodel_2010.txt"))

    faults = []
    for fault_name in discover_faults(data):
        reason = missing_reason(data, fault_name, set(nhm_faults))
        if reason is not None:
            print(f"WARNING: skipping {fault_name}: {reason}")
        else:
            faults.append(fault_name)

    im_periods = discover_periods(data, faults[0])
    db = IMDB.create(
        db_path, periods=im_periods, components=["geom", "rotd50"], db_meta={"dataset_id": "nz_nshm_2010_fault_ims"}
    )
    sites_seen: set[str] = set()

    for fault_name in tqdm(faults, desc="events"):
        srf_dir = data / "source_data" / fault_name / "Srf"
        im_dir = data / "im_data" / fault_name / "IM"
        base = pd.read_csv(srf_dir / f"{fault_name}.csv").iloc[0]

        db.add_events(
            pd.DataFrame(
                [
                    {
                        "event_id": fault_name,
                        "magnitude": base["magnitude"],
                        "tect_type": base["tect_type"],
                        "dip": base["dip"],
                        "dip_dir": base["dip_dir"],
                        "dtop": base["dtop"],
                        "dbottom": base["dbottom"],
                        "length": base["length"],
                        "metadata": json.dumps(
                            {
                                "fault_type": base["fault_type"],
                                "plane_count": int(base["plane_count"]),
                                "slip_rate": base["slip_rate"],
                            }
                        ),
                    }
                ]
            )
        )

        trace = nhm_faults[fault_name].trace[:, ::-1]  # (lon,lat) -> (lat,lon)
        fault = Fault.from_trace_points(
            trace, dtop=base["dtop"], dbottom=base["dbottom"], dip=base["dip"], dip_dir=base["dip_dir"]
        )

        event_sites: set[str] = set()
        realisations, rel_contexts, records, psa_rows = [], [], [], []
        rel_paths = sorted(srf_dir.glob(f"{fault_name}_REL*.csv"), key=rel_number)
        for rel_path in rel_paths:
            rel = pd.read_csv(rel_path).iloc[0]
            rel_id = f"{fault_name}_REL{rel_number(rel_path):02d}"
            x = 0.5 + rel["shypo"] / fault.length
            y = rel["dhypo"] / fault.width
            hypo_lat, hypo_lon, hypo_depth_m = fault.fault_coordinates_to_wgs_depth_coordinates(np.array([x, y]))
            realisations.append(
                {
                    "rel_id": rel_id,
                    "event_id": fault_name,
                    "magnitude": rel["magnitude"],
                    "rake": rel["rake"],
                    "hypo_lat": hypo_lat,
                    "hypo_lon": hypo_lon,
                    "hypo_depth": hypo_depth_m / 1000,
                    "metadata": json.dumps(
                        {
                            "shypo": rel["shypo"],
                            "dhypo": rel["dhypo"],
                            "seed": int(rel["seed"]),
                            "srfgen_seed": int(rel["srfgen_seed"]),
                            "sdrop": rel["sdrop"],
                        }
                    ),
                }
            )

            im = pd.read_csv(im_dir / f"{rel_id}.csv")
            psa_cols = [f"pSA_{p}" for p in im_periods]

            event_sites.update(im["station"])

            records.append(
                pd.DataFrame(
                    {
                        "rel_id": rel_id,
                        "site_id": im["station"],
                        "component": im["component"],
                        "kind": "simulated",
                        **{col: im[col] for col in SCALAR_COLS},
                    }
                )
            )
            psa_rows.append(im[psa_cols].to_numpy())

            rel_contexts.append(
                pd.DataFrame(
                    {
                        "rel_id": rel_id,
                        "site_id": im["station"],
                        "mag": rel["magnitude"],
                        "rake": rel["rake"],
                        "hypo_depth": hypo_depth_m / 1000,
                    }
                )
            )

        new_sites = sorted(event_sites - sites_seen)
        if new_sites:
            db.add_sites(sites_all.loc[new_sites].reset_index(names="site_id"))
            sites_seen.update(new_sites)
        db.add_realisations(pd.DataFrame(realisations))

        # site_event distances, computed once per event, over every site this event references.
        sorted_sites = sorted(event_sites)
        coords = sites_all.loc[sorted_sites]
        latlon = coords[["lat", "lon"]].to_numpy()
        latlondepth = np.column_stack([latlon, np.zeros(len(coords))])
        rrup = fault.rrup_distance(latlondepth)
        rjb = fault.rjb_distance(latlondepth)
        rx, ry = fault.rx_ry_distance(latlon)
        site_event_df = pd.DataFrame(
            {
                "site_id": sorted_sites,
                "event_id": fault_name,
                "rrup": np.asarray(rrup) / 1000,
                "rjb": np.asarray(rjb) / 1000,
                "rx": np.asarray(rx) / 1000,
                "ry": np.asarray(ry) / 1000,
            }
        )
        db.add_site_event(site_event_df)
        db.add_records(pd.concat(records, ignore_index=True), pSA=np.concatenate(psa_rows, axis=0))

        # empirical GMM predictions, over the same (rel_id, site_id) pairs as the simulated records.
        context = pd.concat(rel_contexts, ignore_index=True)
        site_attrs = sites_all.loc[context["site_id"], ["vs30", "z1p0", "z2p5"]].reset_index(drop=True)
        distances = (
            site_event_df.set_index("site_id")
            .loc[context["site_id"], ["rrup", "rjb", "rx", "ry"]]
            .reset_index(drop=True)
        )
        rupture_df = pd.concat([context.reset_index(drop=True), site_attrs, distances], axis=1).rename(
            columns={"z1p0": "z1pt0", "z2p5": "z2pt5"}
        )
        rupture_df["dip"] = base["dip"]
        rupture_df["ztor"] = base["dtop"]
        rupture_df["zbot"] = base["dbottom"]
        rupture_df["vs30measured"] = True
        rupture_df["backarc"] = False

        gmm_df, gmm_psa, gmm_psa_sigma = run_pSA_logic_tree(rupture_df, base["tect_type"], im_periods)
        db.add_records(gmm_df, pSA=gmm_psa, pSA_sigma=gmm_psa_sigma)

    problems = db.validate()
    print("validate():", problems or "clean")
    print("events:", len(db.get_events()))
    print("realisations:", len(db.get_realisations()))
    print("sites:", len(db.get_sites()))
    print("records by kind:")
    print(db.get_records()["kind"].value_counts())
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory containing source_data/, im_data/ and the site files")
    parser.add_argument(
        "db_path", type=Path, help="Output IMDB path"
    )
    args = parser.parse_args()
    main(args.data, args.db_path)
