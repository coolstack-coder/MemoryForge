import numpy as np

from memoryforge.bench import get_task
from memoryforge.memory.embeddings import HashEmbedder
from memoryforge.memory.lessons import leak_reason
from memoryforge.memory.reranker import train_reranker
from memoryforge.memory.retrieval import FEATURES
from memoryforge.memory.store import open_sqlite
from memoryforge.memory.utility import utility


def test_hash_embedder_is_normalised_and_semantic():
    e = HashEmbedder()
    v = e.embed(["handle the empty list first", "when the list is empty return early", "sort ties stably"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_store_round_trip_and_nearest_neighbours():
    store, e = open_sqlite(":memory:"), HashEmbedder()
    texts = ["guard empty inputs before indexing", "use a stable sort for ties"]
    for t in texts:
        v = e.embed([t])[0]
        store.add_lesson(namespace="ns", text=t, pattern="empty_input", embedding=v, context=v,
                         embedder=e.name, status="verified", source_task="x", created_step=0)
    hits = store.candidates("ns", ["verified"], e.embed(["empty input indexing"])[0])
    assert hits[0]["text"] == texts[0]
    assert store.candidates("other", ["verified"], e.embed(["x"])[0]) == []


def test_utility_sign():
    good = utility(alpha=20, beta=2, alpha0=5, beta0=8)
    bad = utility(alpha=2, beta=12, alpha0=8, beta0=4)
    assert good.lift_mean > 0.3 and good.p_positive > 0.95
    assert bad.lift_mean < 0 and bad.p_positive < 0.05


def test_leak_guard():
    task = get_task("count_in_window")
    assert leak_reason("When a range is inclusive, compare with <= on both ends.", task) is None
    assert leak_reason("In count_in_window use <= hi.", task) is not None


def test_reranker_needs_data_and_learns_signal():
    assert train_reranker([]) is None
    rng = np.random.default_rng(0)
    events = []
    for _ in range(400):
        f = {k: float(rng.random()) for k in FEATURES}
        good = f["prior"] > 0.5
        events.append({"features": f, "withheld": False, "passed": bool(good or rng.random() < 0.1),
                       "pattern_hit": not good and rng.random() < 0.8, "score": float(rng.random())})
    model = train_reranker(events, epochs=150)
    assert model is not None and model.val_auc > 0.75
