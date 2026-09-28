"""Drive the unmixing tool's browser half in real headless Chrome and report.

    ../../../venv_for_HSI/bin/python selftest/run.py

Follows the recipe the annotation tool settled on: a throwaway copy of
index.html with a probe appended, a tiny receiver on another port, Chrome on a
real wall clock (never --virtual-time-budget, which races the fetches).

The app server runs against a scratch --out directory, so nothing the probe
paints or unmixes touches Data/Derived.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent
PY = sys.executable
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCRATCH = Path(tempfile.mkdtemp(prefix="unmix-selftest-"))
APP_PORT, RECV_PORT = 8799, 8800

results = []
done = threading.Event()


class Recv(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            rec = json.loads(body)
        except Exception:
            rec = {"raw": body.decode("utf8", "replace")}
        results.append(rec)
        if rec.get("done"):
            done.set()
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


def main():
    out_dir = SCRATCH / "scratch_out"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    probe = (Path(__file__).resolve().parent / "probe.js").read_text().replace("__RECV__", str(RECV_PORT))
    page = TOOL / "static" / "_selftest.html"
    html = (TOOL / "static" / "index.html").read_text()
    page.write_text(html.replace("</body>", f"<script>\n{probe}\n</script>\n</body>"))

    recv = ThreadingHTTPServer(("127.0.0.1", RECV_PORT), Recv)
    threading.Thread(target=recv.serve_forever, daemon=True).start()

    app = subprocess.Popen(
        [PY, "unmix.py", "--port", str(APP_PORT), "--no-browser", "--out", str(out_dir)],
        cwd=TOOL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)

    # A fresh profile path per run: Chrome keeps a singleton lock on a
    # user-data-dir, and reusing (or deleting) one a previous headless run
    # still holds makes the new launch exit without ever loading the page.
    profile = SCRATCH / f"chrome_profile_{os.getpid()}"
    chrome = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", "--window-size=1600,1000",
         f"--user-data-dir={profile}", "--disable-extensions",
         f"http://127.0.0.1:{APP_PORT}/static/_selftest.html"],
        stdout=subprocess.DEVNULL, stderr=open(SCRATCH / "chrome.log", "w"))

    ok = done.wait(420)
    time.sleep(1)
    chrome.kill()
    app.terminate()
    try:
        log = app.communicate(timeout=5)[0]
    except subprocess.TimeoutExpired:
        app.kill()
        log = ""
    page.unlink(missing_ok=True)
    shutil.rmtree(profile, ignore_errors=True)

    fails = 0
    for r in results:
        if r.get("done"):
            continue
        flag = "ok  " if r.get("ok") else "FAIL"
        fails += 0 if r.get("ok") else 1
        print(f"  {flag}  {r.get('name', '?'):46s} {r.get('detail', '')}")
    print(f"\n{len(results) - 1} checks, {fails} failed"
          + ("" if ok else "  (timed out before the probe finished)"))
    if log and ("Traceback" in log or "Error" in log):
        print("\nserver log:\n" + log)
    if len(results) < 2:
        print("\nchrome said:\n" + (SCRATCH / "chrome.log").read_text()[-2000:])
    return 1 if fails or not ok else 0


if __name__ == "__main__":
    sys.exit(main())
