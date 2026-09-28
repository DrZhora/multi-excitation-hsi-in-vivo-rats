"""The linear unmixing tool: a local web app, and the same thing as a batch job.

    cd "Code/Linear_unmixing_tool"
    ../../venv_for_HSI/bin/python unmix.py

opens a browser on `Data/Processed`. Pick a leg, pick the version you want to
look at while you work, draw a region for each structure, and press *Unmix this
leg* — every version of that leg is unmixed from those same regions and scored
against the leg's label mask, and the results land side by side.

    ../../venv_for_HSI/bin/python unmix.py --batch --report

runs every leg that already has regions saved, without a browser, and writes
the comparison page.

The server is deliberately plain: stdlib http.server, one thread per request,
bound to localhost. Everything it knows lives on disk under
`Data/Derived/Linear_unmixing/`, so closing it loses nothing.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

import unmix_lib as U

STATIC = Path(__file__).resolve().parent / "static"

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".png": "image/png", ".json": "application/json",
        ".csv": "text/csv; charset=utf-8", ".svg": "image/svg+xml"}


# =============================================================================
# State the server keeps: the discovered legs, one open cube cache, one job
# =============================================================================
class State:
    """Everything the handlers share. One instance, guarded by its own lock."""

    def __init__(self, data_root, out_root, trees=None):
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        self.trees = trees
        self.lock = threading.Lock()
        self.legs = U.discover(self.data_root, self.out_root, trees)
        self.classes = U.classes_json(self.data_root)
        self._cube = None                  # (version id + leg) -> open memmap
        self._cube_key = None
        self.job = None

    def version(self, leg, vid):
        for v in self.legs[leg]["versions"]:
            if v["id"] == vid:
                return v
        raise KeyError(f"{leg} has no version {vid!r}")

    def cube(self, leg, vid):
        """The one cube currently being looked at, kept mapped between requests.

        Only one is held: the browser asks for band after band of a single
        version while you scrub the slider, and a 100 MB mapping per version
        times 22 would be pointless.
        """
        key = (leg, vid)
        with self.lock:
            if key != self._cube_key:
                self._cube = U.open_cube(self.version(leg, vid))
                self._cube_key = key
            return self._cube

    def drop_cube(self):
        with self.lock:
            self._cube, self._cube_key = None, None

    def leg_payload(self, leg):
        entry = self.legs[leg]
        sel = U.load_selection(leg, self.out_root)
        rois = U.load_rois(leg, (entry["Y"], entry["X"]), self.out_root)
        structures = (sel or {}).get("structures") or U.default_structures(self.data_root)
        results = U.leg_dir(leg, self.out_root) / "metrics.csv"
        return {
            "leg": leg, "Y": entry["Y"], "X": entry["X"],
            "warnings": entry["warnings"],
            "versions": [{k: v[k] for k in
                          ("id", "label", "group", "tree", "kind", "key", "C",
                           "has_mask", "duplicate_of")} | {"name": v["path"].name}
                         for v in entry["versions"]],
            "selection": sel or {"reference": entry["versions"][0]["id"],
                                 "structures": structures,
                                 "solver": dict(U.DEFAULT_SOLVER)},
            "roi_counts": [int((rois == i + 1).sum()) for i in range(len(structures))],
            "has_results": results.exists(),
        }


# =============================================================================
# The unmixing job — one at a time, polled by the browser
# =============================================================================
class Job:
    def __init__(self, legs):
        self.legs = legs
        self.done_legs = []
        self.leg = legs[0] if legs else ""
        self.i, self.n, self.version, self.stage = 0, 1, "", "starting"
        self.error = None
        self.finished = False
        self.cancel = False

    def snapshot(self):
        return {"legs": self.legs, "done_legs": self.done_legs, "leg": self.leg,
                "i": self.i, "n": self.n, "version": self.version,
                "stage": self.stage, "finished": self.finished, "error": self.error}


def start_job(state, legs, save_npy=False):
    job = Job(legs)
    state.job = job

    def progress(i, n, vid, stage):
        job.i, job.n, job.version, job.stage = i, n, vid, stage

    def run():
        try:
            for leg in legs:
                if job.cancel:
                    break
                job.leg = leg
                U.run_leg(leg, state.legs, state.out_root, save_npy,
                          progress, lambda: job.cancel, state.data_root)
                job.done_legs.append(leg)
            # Rebuild the flat table and the page from whatever is on disk, so
            # a run of one leg still refreshes the cross-leg comparison.
            rows = U.collect_all(state.out_root)
            if rows:
                U.write_report(state.out_root, rows, title=_report_title(state))
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.stage = "cancelled" if job.cancel else "done"
            job.finished = True

    threading.Thread(target=run, daemon=True).start()
    return job


# =============================================================================
# HTTP
# =============================================================================
class Handler(BaseHTTPRequestHandler):
    state: State = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                        # the console is for progress

    # -- plumbing ------------------------------------------------------------
    def _send(self, body, ctype="application/json", code=200, cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, exc, code=400):
        self._send({"error": f"{type(exc).__name__}: {exc}"}, code=code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n)

    def _q(self, url):
        return {k: v[0] for k, v in parse_qs(url.query).items()}

    def _file(self, path, root):
        path = path.resolve()
        if root.resolve() not in path.parents or not path.is_file():
            return self._send({"error": "not found"}, code=404)
        # Nothing here is cached. The result images are the trap: their URLs
        # are stable (`/out/<leg>/<version>/classmap.png`) but their contents
        # change every run, so a cached copy shows the *previous* run's regions
        # and class maps while the numbers beside them are current. Everything
        # is a local disk read, so re-fetching costs nothing worth having.
        self._send(path.read_bytes(), MIME.get(path.suffix, "application/octet-stream"))

    # -- GET -----------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        p, q = unquote(url.path), self._q(url)
        try:
            if p in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), MIME[".html"])
            if p.startswith("/static/"):
                return self._file(STATIC / p[len("/static/"):], STATIC)
            if p.startswith("/out/"):
                return self._file(self.state.out_root / p[len("/out/"):],
                                  self.state.out_root)

            if p == "/api/state":
                return self._send({
                    "data_root": str(self.state.data_root),
                    "out_root": str(self.state.out_root),
                    "classes": self.state.classes,
                    "solver_defaults": U.DEFAULT_SOLVER,
                    "legs": [{"leg": leg,
                              "versions": len(e["versions"]),
                              "X": e["X"], "Y": e["Y"],
                              "has_selection": U.selection_path(
                                  leg, self.state.out_root).exists(),
                              "has_results": (U.leg_dir(leg, self.state.out_root)
                                              / "metrics.csv").exists()}
                             for leg, e in sorted(self.state.legs.items())]})

            if p == "/api/leg":
                return self._send(self.state.leg_payload(q["leg"]))

            if p == "/api/view":
                return self._send(self._render(q), "image/png")

            if p == "/api/channels":
                v = self.state.version(q["leg"], q["version"])
                exc, emi = U._channel_table(v)
                return self._send({"excitation": exc, "emission_nm": emi}, cache=True)

            if p == "/api/mask":
                v = self.state.version(q["leg"], q["version"])
                if not v["has_mask"]:
                    return self._send({"error": "no mask"}, code=404)
                mask = np.array(Image.open(v["mask"]))
                palette = {int(c["id"]): c["color"] for c in self.state.classes["classes"]}
                return self._send(U.png_bytes(U.colour_map(mask, palette), "RGB"),
                                  "image/png")

            if p == "/api/rois":
                e = self.state.legs[q["leg"]]
                rois = U.load_rois(q["leg"], (e["Y"], e["X"]), self.state.out_root)
                return self._send(rois.tobytes(), "application/octet-stream")

            if p == "/api/spectra":
                return self._send(self._spectra(q))

            if p == "/api/results_all":
                rows = U.read_all_rows(self.state.out_root)
                if not rows:
                    return self._send({"error": "nothing has been unmixed yet"},
                                      code=404)
                return self._send({"rows": rows,
                                   "classes": [c["name"] for c in
                                               self.state.classes["classes"]]})

            if p == "/api/results":
                path = U.leg_dir(q["leg"], self.state.out_root) / "metrics.json"
                if not path.exists():
                    return self._send({"error": "not run yet"}, code=404)
                return self._send(path.read_bytes(), MIME[".json"])

            if p == "/api/job":
                job = self.state.job
                return self._send(job.snapshot() if job else {"idle": True})

            return self._send({"error": f"no route {p}"}, code=404)
        except KeyError as exc:
            self._fail(exc, 404)
        except ValueError as exc:
            self._fail(exc, 400)
        except Exception as exc:                     # noqa: BLE001 — report, don't die
            traceback.print_exc()
            self._fail(exc, 500)

    # -- POST ----------------------------------------------------------------
    def do_POST(self):
        url = urlparse(self.path)
        p, q = unquote(url.path), self._q(url)
        try:
            if p == "/api/selection":
                sel = json.loads(self._body())
                return self._send(U.save_selection(q["leg"], sel, self.state.out_root))

            if p == "/api/rois":
                e = self.state.legs[q["leg"]]
                raw = self._body()
                if len(raw) != e["Y"] * e["X"]:
                    raise ValueError(f"expected {e['Y'] * e['X']} bytes, got {len(raw)}")
                U.save_rois(q["leg"],
                            np.frombuffer(raw, np.uint8).reshape(e["Y"], e["X"]),
                            self.state.out_root)
                return self._send({"ok": True})

            if p == "/api/seed":
                leg = q["leg"]
                e = self.state.legs[leg]
                v = next((v for v in e["versions"] if v["has_mask"]), None)
                if v is None:
                    raise ValueError("this leg has no label mask to seed from")
                # The structures travel with the request. Reading them back off
                # disk would seed against whatever the last *saved* selection
                # said, and saving is debounced — rename or drop a structure and
                # seed straight away and the two disagree.
                body = json.loads(self._body() or b"{}")
                sel = U.load_selection(leg, self.state.out_root)
                structures = (body.get("structures") or (sel or {}).get("structures")
                              or U.default_structures(self.state.data_root))
                rois = U.seed_rois_from_mask(np.array(Image.open(v["mask"])), structures,
                                             fraction=float(q.get("fraction", 0.02)))
                return self._send(rois.tobytes(), "application/octet-stream")

            if p == "/api/run":
                body = json.loads(self._body() or b"{}")
                legs = body.get("legs") or [body["leg"]]
                if self.state.job and not self.state.job.finished:
                    raise ValueError("a run is already going")
                missing = [lg for lg in legs
                           if U.load_selection(lg, self.state.out_root) is None]
                if missing:
                    raise ValueError(f"no regions saved for {', '.join(missing)}")
                start_job(self.state, legs, bool(body.get("save_npy")))
                return self._send({"started": legs})

            if p == "/api/cancel":
                if self.state.job:
                    self.state.job.cancel = True
                return self._send({"ok": True})

            if p == "/api/report":
                return self._send({"path": str(U.write_report(self.state.out_root))})

            return self._send({"error": f"no route {p}"}, code=404)
        except KeyError as exc:
            self._fail(exc, 404)
        except ValueError as exc:
            self._fail(exc, 400)
        except Exception as exc:                     # noqa: BLE001
            traceback.print_exc()
            self._fail(exc, 500)

    # -- the two handlers that need real work --------------------------------
    def _render(self, q):
        cube = self.state.cube(q["leg"], q["version"])
        rgb = [int(x) for x in q.get("rgb", "0,0,0").split(",")]
        return U.render_cube(cube, mode=q.get("mode", "mean"),
                             band=int(q.get("band", 0)), rgb=rgb,
                             lo=float(q.get("lo", 1)), hi=float(q.get("hi", 99)),
                             gamma=float(q.get("gamma", 1)))

    def _spectra(self, q):
        """The endmembers the current regions would give on one version.

        Computed live rather than read back from a run, because the point is to
        watch a spectrum settle down while you are still drawing the region.
        """
        leg, vid = q["leg"], q["version"]
        e = self.state.legs[leg]
        sel = U.load_selection(leg, self.state.out_root) or {}
        structures = sel.get("structures") or U.default_structures(self.state.data_root)
        solver = U.DEFAULT_SOLVER | sel.get("solver", {})
        rois = U.load_rois(leg, (e["Y"], e["X"]), self.state.out_root)
        cube = self.state.cube(leg, vid)
        E, counts, extras = U.endmembers(cube, rois, len(structures),
                                         stat=solver["stat"], extra=solver["extra"],
                                         normalize=solver["normalize_endmembers"])
        exc, emi = U._channel_table(self.state.version(leg, vid))
        return {"names": [s["name"] for s in structures] + extras,
                "colors": [s["color"] for s in structures] + ["#8a8f99"] * len(extras),
                "counts": counts, "excitation": exc, "emission_nm": emi,
                "spectra": [[float(x) for x in E[:, k]] for k in range(E.shape[1])]}


# =============================================================================
# Entry points
# =============================================================================
def free_port(preferred):
    """The preferred port if the server could bind it, otherwise any free one.

    The probe has to set SO_REUSEADDR because the server does (HTTPServer sets
    it): without it a port still in TIME_WAIT from the previous run looks taken,
    and restarting the tool would silently move it to a random port while the
    browser tab you already have open keeps pointing at the old one.
    """
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


DEFAULT_PORT = 8765


def serve(state, port, open_browser=True):
    serve._asked = port
    port = free_port(port)
    Handler.state = state
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    # flush: the banner carries the URL, and when stdout is a pipe or a log file
    # it would otherwise sit in the buffer for the life of the server.
    print(f"linear unmixing tool  →  {url}")
    print(f"  cubes   {state.data_root}")
    print(f"  results {state.out_root}")
    print(f"  {len(state.legs)} legs, "
          f"{sum(len(e['versions']) for e in state.legs.values())} cube versions"
          + (f"  (trees: {', '.join(state.trees)})" if state.trees else ""),
          flush=True)
    if port != DEFAULT_PORT and port != getattr(serve, "_asked", port):
        print(f"  (port {getattr(serve, '_asked')} was busy)", flush=True)
    if open_browser:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def _report_title(state):
    """Name the report after what it actually compared."""
    if not state.trees:
        return "Linear unmixing — all legs, both trees"
    names = [t["label"] for t in U.TREES if t["key"] in state.trees]
    return f"Linear unmixing — all legs, {' + '.join(names)}"


def batch(state, only, seed_missing, save_npy, report):
    legs = state.legs
    if seed_missing:
        structures = U.default_structures(state.data_root)
        for leg, e in legs.items():
            if only and leg not in only:
                continue
            if U.load_selection(leg, state.out_root) is not None:
                continue
            v = next((v for v in e["versions"] if v["has_mask"]), None)
            if v is None:
                print(f"  {leg}: no mask, cannot seed")
                continue
            rois = U.seed_rois_from_mask(np.array(Image.open(v["mask"])), structures)
            U.save_rois(leg, rois, state.out_root)
            U.save_selection(leg, {"reference": e["versions"][0]["id"],
                                   "structures": structures,
                                   "solver": dict(U.DEFAULT_SOLVER),
                                   "seeded_from_mask": True}, state.out_root)
            print(f"  {leg}: regions seeded from the label mask")

    # Overwrite one line on a terminal; one line per step when piped to a file,
    # where a carriage return would collapse the whole run into one line.
    tty = sys.stdout.isatty()

    def progress(i, n, vid, stage):
        print(f"\r  {i + 1:3d}/{n} {vid:24s} {stage:9s}",
              end="" if tty else "\n", flush=True)

    rows = U.run_all(legs, state.out_root, save_npy, progress, only=only,
                     root=state.data_root, trees=state.trees)
    print()
    if not rows:
        print("nothing to run — no leg has regions saved yet "
              "(draw some, or pass --seed-missing)")
        return
    print(f"{len(rows)} version runs over "
          f"{len({r['leg'] for r in rows})} legs → {state.out_root}/metrics_all.csv")
    if report:
        print("report:", U.write_report(state.out_root, rows,
                                        title=_report_title(state)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", nargs="?", default=str(U.PROCESSED_ROOT),
                    help="the Data/Processed folder to work on")
    ap.add_argument("--out", default=str(U.DERIVED_ROOT),
                    help="where regions, abundances and metrics are written")
    ap.add_argument("--tree", action="append", choices=[t["key"] for t in U.TREES],
                    help="compare only this tree: 'all' = "
                         "Alligned_Cropped_All_channels_kept, 'oe' = "
                         "Alligned_cropped_overexposed_channels_removed. "
                         "Repeatable; the default is both.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--batch", action="store_true",
                    help="run every leg that has regions saved, then exit")
    ap.add_argument("--leg", action="append",
                    help="limit --batch to this leg (repeatable)")
    ap.add_argument("--seed-missing", action="store_true",
                    help="with --batch: give any leg without regions a set "
                         "sampled out of its label mask")
    ap.add_argument("--save-npy", action="store_true",
                    help="also write the raw float32 abundance stacks")
    ap.add_argument("--report", action="store_true",
                    help="with --batch: write report.html at the end")
    args = ap.parse_args(argv)

    state = State(args.data, args.out, args.tree)
    if not state.legs:
        print(f"no aligned legs found under {args.data}", file=sys.stderr)
        return 1
    if args.batch:
        batch(state, set(args.leg) if args.leg else None,
              args.seed_missing, args.save_npy, args.report)
        return 0
    serve(state, args.port, not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
