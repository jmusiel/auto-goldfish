# web/ -- Flask Web UI

Flask application serving the deck management dashboard, simulation configuration, and client-side simulation infrastructure.

## Structure

```
web/
├── __init__.py              # App factory (create_app), DB init, blueprint registration
├── routes/
│   ├── dashboard.py         # GET / -- deck listing
│   ├── decks.py             # Deck import (Archidekt) and card view
│   ├── simulation.py        # Simulation config page and JSON APIs
│   ├── mana_model.py        # Hypergeometric land count recommender
│   └── runs.py              # /runs page: persisted simulation runs + calibration badge
├── services/
│   └── simulation_runner.py # SimJob + SimulationRunner (background threads)
├── templates/
│   ├── base.html            # Base layout
│   ├── dashboard.html       # Deck list
│   ├── import.html          # Archidekt import form
│   ├── deck_view.html       # Card list grouped by category
│   ├── simulate.html        # Config form + Pyodide simulation client
│   ├── results.html         # Standalone results page
│   └── partials/            # HTMX fragments (job_status, results_content, validation_error)
├── wizard.py                # Card labeling wizard prioritization logic
└── static/
    ├── style.css
    └── js/
        ├── pyodide_worker.js   # Web Worker: loads Pyodide, runs simulation
        ├── client_results.js   # Renders results tables, charts, replay viewer
        ├── deck_store.js       # localStorage CRUD for deck data
        └── labeler_wizard.js   # Pure logic for card labeler decision tree (testable)
```

## How Simulation Works

All simulation runs client-side via Pyodide (CPython in WebAssembly):

1. Page loads `simulate.html`, which initializes a Web Worker (`pyodide_worker.js`)
2. Worker downloads the `auto_goldfish` wheel from `/sim/api/wheel/<filename>` and installs it into Pyodide
3. On form submit, the main thread fetches deck data (`/sim/api/<deck>/deck`) and effects (`/sim/api/<deck>/effects`), then posts to the worker
4. Worker runs `pyodide_runner.run_simulation()`, sends progress updates back
5. On completion, `client_results.js` renders results inline; a fire-and-forget POST to `/sim/api/<deck>/results` persists to the database (if configured)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sim/<deck>` | Config page with effect editor |
| POST | `/sim/<deck>/overrides` | Save card effect overrides |
| POST | `/sim/<deck>/annotate` | Save card annotation (fire-and-forget, DB optional) |
| GET | `/sim/api/<deck>/deck` | Deck card list (JSON) |
| GET | `/sim/api/<deck>/effects` | Merged effect overrides + registry (JSON) |
| POST | `/sim/api/<deck>/results` | Persist client-side simulation results |
| GET | `/sim/api/wheel` | Latest wheel filename |
| GET | `/sim/api/wheel/<filename>` | Serve wheel file |

## Configuration

- `SECRET_KEY` env var (defaults to `"dev"`)
- `DATABASE_URL` env var -- if set, enables Postgres persistence via `db/` module
- `AUTO_GOLDFISH_CALIBRATE` env var -- defaults to enabled. When the DB is reachable and contains persisted raw composite stats, the `/runs` page tunes the 1-10 score anchors against the empirical distribution (Bayesian-shrunk toward defaults). Set to `0` to fall back to built-in default anchors. The `/runs` page shows a "Calibrated" or "Default anchors" badge with the active values. The `/sim/<deck>` page injects the active anchors into `window.CASTER_CALIBRATION` so `client_results.js` can render the "What does my CASTER Score mean?" expandable panel against current calibration.
