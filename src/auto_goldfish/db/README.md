# db/ -- Optional Postgres Persistence

SQLAlchemy 2.0 database layer for persisting simulation results and deck card labels. Entirely optional -- if `DATABASE_URL` is not set, the app runs without a database.

## Structure

```
db/
├── __init__.py      # Package docstring
├── models.py        # SQLAlchemy ORM models (8 tables)
├── session.py       # Engine creation, session context manager
└── persistence.py   # Get-or-create helpers, save functions, convenience wrappers
```

## Schema

```
CardRow              -- canonical card names + cached metadata (id, name, types_json, cmc)
EffectLabelRow       -- deduplicated effect JSON blobs (id, effects_json)
DeckRow              -- saved decks (id, name, created_at)
DeckCardRow          -- deck <-> card join: effect label, user_edited, quantity, is_commander
SimulationRunRow     -- one simulation run (job_id, config params, optimal_land_count)
SimulationResultRow  -- per-land-count stats (mean_mana, consistency, CIs, percentiles)
CardPerformanceRow   -- bottom 10 archetype pools with effects at optimal land count (stored against the example card)
CalibrationCacheRow  -- single-row cache of the most recent CASTER anchors, keyed by simulation_results row count
```

`CardRow.types_json` (JSON list, e.g. `["Creature", "Elf"]`) and `CardRow.cmc`
are populated whenever a deck is opened in the config page. Together with
`DeckCardRow.quantity` and `DeckCardRow.is_commander`, they let the runs
page reconstruct full deck composition (commanders, mana curve, land
count, ramp/draw breakdowns) without reading deck JSON from disk -- which
matters on Vercel where deck files don't ship to the runtime.

By default `init_db()` runs `Base.metadata.create_all()` plus `_migrate()` at startup so the schema is created on first use. Set `AUTO_GOLDFISH_SKIP_MIGRATE=1` (production / Vercel) to skip both -- migrations are then expected to have already run during the deploy build via `scripts/migrate.py`.

## Usage

The web layer calls into this module at three points:

1. **Deck config page** (`/sim/<deck>`) calls `persist_deck_cards()` to save card labels and overrides
2. **SimulationRunner** calls `persist_completed_job()` after a server-side simulation completes
3. **Client results API** (`POST /sim/api/<deck>/results`) calls `save_simulation_run()` to persist Pyodide results

All calls are wrapped in try/except so database failures never break the app.
The client results endpoint distinguishes between "DB not configured" (returns
200 with `persisted: false`) and "DB configured but the write failed" (returns
500 with the error string), so the browser can surface real persistence
failures instead of silently dropping them.

## Serverless considerations

`init_db()` configures SQLAlchemy with `poolclass=NullPool` and
`pool_pre_ping=True` because the app is deployed on Vercel, where workers
freeze between requests. A long-lived SQLAlchemy connection pool can hold
sockets that the upstream proxy has already torn down by the time the worker
thaws; the next request then sees an exception. Letting Neon's own pgbouncer
do the pooling and opening a fresh connection per session keeps things sane.

### Migrations at deploy time, not at request time

Cold-start workers must not race on `ALTER TABLE`. Instead:

* Build step: `scripts/vercel_build.sh` runs `python scripts/migrate.py`,
  which calls `init_db(..., run_migrations=True)` once with the deploy's
  `DATABASE_URL`. All `create_all` + `_migrate` work happens here.
* Runtime: Vercel sets `AUTO_GOLDFISH_SKIP_MIGRATE=1` so each cold-start
  worker calls `init_db()` with `run_migrations=False`, opening a fresh
  engine and running `verify_schema()` (read-only) to log loudly if the
  prod DB is missing any required column.

Local dev/tests leave `AUTO_GOLDFISH_SKIP_MIGRATE` unset, preserving the
auto-create behaviour expected by `tests/` and ad-hoc `flask run` workflows.

### Calibration cache (CalibrationCacheRow)

`metrics/calibration.py` reads/writes a single row in `calibration_cache`
keyed by the current `simulation_results` row count. A freshly-thawed
worker that finds a matching row deserializes the stored anchors instead
of running the percentile + Bayesian-shrinkage pass. Adding new sim rows
naturally invalidates the cache (row count differs); the next caller
recomputes and writes the new row back. Concurrent recomputes converge to
the same answer for a given row count, so a last-writer-wins merge by
primary key (`id=1`) is sufficient.

## Setup

```bash
uv sync --extra db
DATABASE_URL="postgresql://user:pass@host/dbname" .venv/bin/flask --app src.auto_goldfish.web:create_app run
```
