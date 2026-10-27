import json

from fastapi.testclient import TestClient

from memoryforge.eval.experiment import run_experiment
from memoryforge.memory.store import open_sqlite


def test_small_experiment_and_dashboard(tmp_path, monkeypatch):
    store = open_sqlite(tmp_path / "mf.sqlite3")
    res = run_experiment(seeds=1, n_train=6, n_eval=3, store=store, embedder="hash", out_dir=tmp_path)
    run_id = res["meta"]["run_id"]
    assert set(res["summary"]) == {"none", "reflexion", "raw_memory", "verified", "verified_rerank", "shuffled"}
    assert res["meta"]["simulated"] is True
    saved = json.loads((tmp_path / run_id / "results.json").read_text())
    assert saved["meta"]["status"] == "done"
    assert any(r["phase"] == "eval" for r in saved["rows"])

    from memoryforge.server import app as server

    monkeypatch.setattr(server.settings, "runs_dir", tmp_path)
    monkeypatch.setattr(server, "_store", store)
    client = TestClient(server.app)
    assert client.get("/").status_code == 200
    assert client.get("/api/runs").json()[0]["run_id"] == run_id
    assert "summary" in client.get(f"/api/runs/{run_id}").json()
    lessons = client.get(f"/api/runs/{run_id}/lessons", params={"memory": "verified"}).json()
    assert lessons["total"] == sum(lessons["counts"].values())
    att = client.get(f"/api/runs/{run_id}/attempts",
                     params={"task_id": "free_ranges", "condition": "verified", "seed": 0}).json()
    assert att["trials"] and att["task"]["id"] == "free_ranges"
