# db/ -- Optional Postgres Persistence

SQLAlchemy 2.0 database layer for persisting simulation results and deck card labels. Entirely optional -- if `DATABASE_URL` is not set, the app runs without a database.

## Structure

```
db/
├── __init__.py      # Package docstring
├── models.py        # SQLAlchemy ORM models (7 tables)
├── session.py       # Engine creation, session context manager
└── persistence.py   # Get-or-create helpers, save functions, convenience wrappers
```

## Schema

```
CardRow              -- canonical card names (id, name)
EffectLabelRow       -- deduplicated effect JSON blobs (id, effects_json)
DeckRow              -- saved decks (id, name, created_at)
DeckCardRow          -- deck <-> card join with effect label + user_edited flag
SimulationRunRow     -- one simulation run (job_id, config params, optimal_land_count)
SimulationResultRow  -- per-land-count stats (mean_mana, consistency, CIs, percentiles)
CardPerformanceRow   -- bottom 10 archetype pools with effects at optimal land count (stored against the example card)
```

Tables are created automatically via `init_db()` on app startup.

## Usage

The web layer calls into this module at three points:

1. **Deck config page** (`/sim/<deck>`) calls `persist_deck_cards()` to save card labels and overrides
2. **SimulationRunner** calls `persist_completed_job()` after a server-side simulation completes
3. **Client results API** (`POST /sim/api/<deck>/results`) calls `save_simulation_run()` to persist Pyodide results

All calls are wrapped in try/except so database failures never break the app.

## Setup

```bash
uv sync --extra db
DATABASE_URL="postgresql://user:pass@host/dbname" .venv/bin/flask --app src.auto_goldfish.web:create_app run
```
