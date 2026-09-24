import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from http.server import ThreadingHTTPServer
from scripts.imdb_progress import handler_for, snapshot, write_stage


def test_progress_counts_only_committed_new_cells(tmp_path):
    run, state = tmp_path / "run", tmp_path / "state.json"
    assert snapshot(run, state, 3600)["stage"] == "waiting"
    cells = run / "cells"
    cells.mkdir(parents=True)
    (cells / "one.json").write_text("{}")
    (cells / "one.npz").write_bytes(b"")
    (cells / "incomplete.tmp").write_text("{}")
    write_stage(state, "replaying")
    result = snapshot(run, state, 3600)
    assert result["completed"] == 1 and result["total"] == 3600
    assert result["stage"] == "replaying"
    write_stage(state, "complete")
    with pytest.raises(ValueError, match="inconsistent"):
        snapshot(run, state, 3600)
    assert snapshot(run, state, 1)["percent"] == 100


def test_server_updates_without_serving_private_files(tmp_path):
    state = tmp_path / "state.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(tmp_path / "run", state, 3600))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert b"<progress" in response.read()
        write_stage(state, "fitting")
        with urlopen(base + "/status") as response:
            assert json.load(response)["stage"] == "fitting"
            assert response.headers["Cache-Control"] == "no-store"
        write_stage(state, "validating")
        with urlopen(base + "/status") as response:
            assert json.load(response)["stage"] == "validating"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/state.json")
        assert error.value.code == 404
        state.write_text("not json")
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/status")
        assert error.value.code == 503
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
