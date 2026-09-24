"""Small loopback-only progress display for the added PCA experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


STAGES = {
    "waiting": "Waiting to start",
    "fitting": "Fitting PCA",
    "replaying": "Running PCA replays",
    "validating": "Checking results and updating the report",
    "complete": "Complete",
    "failed": "Stopped with an error",
}


def write_stage(path: Path, stage: str) -> None:
    if stage not in STAGES:
        raise ValueError("unknown progress stage")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot(run: Path, state_path: Path, total: int) -> dict:
    if total <= 0:
        raise ValueError("total must be positive")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"stage": "waiting"}
    stage = state["stage"]
    if stage not in STAGES:
        raise ValueError("unrecognized progress stage")
    completed = sum(1 for _ in (run / "cells").glob("*.json"))
    if completed > total or (stage == "complete" and completed != total):
        raise ValueError("progress stage/count is inconsistent")
    return {
        "stage": stage, "status": STAGES[stage], "completed": completed,
        "total": total, "percent": round(100 * completed / total, 1),
        "note": "Saved PCA settings only; the 3,600 original results are preserved separately.",
    }


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PCA experiment progress</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f3;color:#172d39;font:16px/1.5 system-ui,Arial,sans-serif;display:grid;min-height:100vh;place-items:center}
main{width:min(580px,calc(100% - 32px));padding:30px;background:white;border:1px solid #d4dfdf;border-radius:14px}
h1{font-size:24px;margin:0 0 10px}#percent{font-size:48px;font-weight:750;margin:0}
progress{display:block;width:100%;height:24px;accent-color:#007d7c;margin:14px 0}
p{margin:10px 0}.muted{font-size:14px;color:#49616d}#status{font-weight:650}#connection{color:#a13813}
</style></head><body><main><h1>PCA experiment</h1><p id="percent">Loading...</p>
<progress id="bar" value="0" max="3600" aria-label="Completed PCA settings"></progress>
<p id="count"></p><p id="status" role="status" aria-live="polite"></p><p id="connection" role="alert"></p>
<p class="muted">Updates automatically every 5 seconds.</p><p class="muted" id="note"></p></main>
<script>
const $=id=>document.getElementById(id);
async function update(){
try{const response=await fetch("/status",{cache:"no-store"});const s=await response.json();if(!response.ok)throw new Error(s.error||"Progress is unavailable");
$("bar").max=s.total;$("bar").value=s.completed;$("percent").textContent=s.percent.toFixed(1)+"%";
$("count").textContent=s.completed.toLocaleString()+" / "+s.total.toLocaleString()+" settings saved";
$("status").textContent=s.status;$("note").textContent=s.note;$("connection").textContent="";
}catch(error){$("connection").textContent="Progress updates unavailable: "+error.message+". Last displayed count is unchanged."}
}
update();setInterval(update,5000);
</script></body></html>"""


def handler_for(run: Path, state: Path, total: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                code, content_type, content = 200, "text/html; charset=utf-8", PAGE.encode()
            elif path == "/status":
                try:
                    content = json.dumps(snapshot(run, state, total)).encode()
                    code = 200
                except (OSError, ValueError, KeyError) as exc:
                    content, code = json.dumps({"error": str(exc)}).encode(), 503
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args) -> None:
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(format, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs_imdb/private_runs/imdb-pca-40-replay"))
    parser.add_argument("--state", type=Path, default=Path("outputs_imdb/private_runs/imdb-pca-progress.json"))
    parser.add_argument("--total", type=int, default=3600)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--stage", choices=list(STAGES), help="Record a workflow stage and exit instead of serving.")
    args = parser.parse_args()
    if args.total <= 0 or not 0 <= args.port <= 65535:
        parser.error("invalid total or port")
    if args.stage:
        write_stage(args.state, args.stage)
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.run, args.state, args.total))
    print(f"PCA progress: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
