-- MemoryForge schema for PostgreSQL 16 + pgvector.
-- A "namespace" isolates the memory of one (experiment, condition, seed) so
-- conditions never leak lessons into each other.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    meta        JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id      BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    namespace       TEXT NOT NULL,
    condition       TEXT NOT NULL,
    seed            INT NOT NULL,
    phase           TEXT NOT NULL,            -- train | eval | replay
    step            INT NOT NULL,             -- position in the task stream
    task_id         TEXT NOT NULL,
    trial           INT NOT NULL,
    code            TEXT NOT NULL,
    passed          BOOLEAN NOT NULL,
    n_passed        INT NOT NULL,
    n_tests         INT NOT NULL,
    feedback        TEXT,
    pattern         TEXT,                     -- classifier diagnosis
    pattern_conf    DOUBLE PRECISION,
    true_pattern    TEXT,                     -- ground truth, simulator only
    reflection      TEXT,
    retrieved       JSONB NOT NULL DEFAULT '[]',
    tokens          INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS attempts_run_idx ON attempts (run_id, condition, phase);
CREATE INDEX IF NOT EXISTS attempts_task_idx ON attempts (task_id);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id        BIGSERIAL PRIMARY KEY,
    namespace        TEXT NOT NULL,
    text             TEXT NOT NULL,
    pattern          TEXT NOT NULL,
    embedding        vector(384) NOT NULL,    -- embedding of the lesson text
    context          vector(384) NOT NULL,    -- embedding of the task it came from
    embedder         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'candidate'
                     CHECK (status IN ('candidate', 'verified', 'deprecated', 'merged', 'rejected')),
    source_task      TEXT NOT NULL,
    source_attempt   BIGINT,
    created_step     INT NOT NULL DEFAULT 0,
    alpha            DOUBLE PRECISION NOT NULL DEFAULT 1,  -- injected: pattern avoided
    beta             DOUBLE PRECISION NOT NULL DEFAULT 1,  -- injected: pattern occurred
    alpha0           DOUBLE PRECISION NOT NULL DEFAULT 1,  -- withheld: pattern avoided
    beta0            DOUBLE PRECISION NOT NULL DEFAULT 1,  -- withheld: pattern occurred
    uses             INT NOT NULL DEFAULT 0,
    replay           JSONB NOT NULL DEFAULT '{}',
    history          JSONB NOT NULL DEFAULT '[]',
    merged_into      BIGINT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS lessons_ns_idx ON lessons (namespace, status);
CREATE INDEX IF NOT EXISTS lessons_emb_hnsw ON lessons USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS lessons_ctx_hnsw ON lessons USING hnsw (context vector_cosine_ops);

CREATE TABLE IF NOT EXISTS retrieval_events (
    event_id     BIGSERIAL PRIMARY KEY,
    attempt_id   BIGINT NOT NULL REFERENCES attempts ON DELETE CASCADE,
    lesson_id    BIGINT NOT NULL REFERENCES lessons ON DELETE CASCADE,
    namespace    TEXT NOT NULL,
    rank         INT NOT NULL,
    score        DOUBLE PRECISION NOT NULL,
    features     JSONB NOT NULL,
    withheld     BOOLEAN NOT NULL DEFAULT FALSE,
    passed       BOOLEAN,
    pattern_hit  BOOLEAN
);
CREATE INDEX IF NOT EXISTS retrieval_ns_idx ON retrieval_events (namespace);
