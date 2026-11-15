"""Build graph.db: run a source's reader over its files in parallel and store the graphs.

uv run ingest.py <source> [item ...]   # items: file stems, or paths under the source for opening_design
"""
import os
import sys
import traceback
from multiprocessing import Pool

import graph
import swiss
import cubicasa
import ifc
import scans
import sheets

READERS = {"swiss_dwellings": swiss, "cubicasa5k": cubicasa, "ifc_models": ifc, "ifc_bench": ifc,
           "opening_design": sheets, **{s: scans for s in scans.SOURCES}}


def build(job):
    source, item = job
    try:
        return READERS[source].read(item)
    except Exception:
        print(f"{item}: {traceback.format_exc()}", file=sys.stderr)


def main(source, *names):
    reader = READERS[source]
    items = reader.items(source) if reader in (ifc, scans) else reader.items()
    if names:
        items = [i for i in items if (i[1] if isinstance(i, tuple) else i) in names]
    jobs = [(source, i) for i in items]
    con = graph.connect()
    # IFC's native parser retains sizeable models; recycle workers to release that memory.
    workers = 1 if source == "ifc_bench" else 2 if reader is ifc else os.cpu_count() - 1
    with Pool(workers, maxtasksperchild=1 if reader is ifc else None) as pool:
        for n, g in enumerate(pool.imap_unordered(build, jobs, chunksize=1 if reader is ifc else 2), 1):
            if g:
                graph.save(con, g)
                if reader is ifc:
                    con.commit()
            if n % 100 == 0:
                con.commit()
                print(f"{n}/{len(jobs)}", flush=True)
    con.commit()


if __name__ == "__main__":
    main(*sys.argv[1:])
