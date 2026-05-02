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
│   └── runs.py              # /runs page: 20 most recent runs + top/bottom-by-stat views and per-run deck breakdown
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
5. On completion, `client_results.js` renders results inline and POSTs to `/sim/api/<deck>/results`. The endpoint returns 200 with `persisted: false` when no `DATABASE_URL` is configured, 200 with `persisted: true` on a successful write, and 500 with an `error` field when persistence is configured but the write fails — so client-side code can distinguish "no DB" from a real outage.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sim/<deck>` | Config page with effect editor |
| POST | `/sim/<deck>/overrides` | Save card effect overrides |
| POST | `/sim/<deck>/annotate` | Save card annotation (fire-and-forget, DB optional) |
| GET | `/sim/api/<deck>/deck` | Deck card list (JSON) |
| GET | `/sim/api/<deck>/effects` | Merged effect overrides + registry (JSON) |
| POST | `/sim/api/<deck>/results` | Persist client-side simulation results |
| GET | `/sim/api/wheel` | Latest wheel filename + mtime (for cache-busting) |
| GET | `/sim/api/wheel/<filename>` | Serve wheel file |
| GET | `/runs/` | Renders the 20 most recent simulation runs |
| GET | `/runs/api/data?view=recent\|top\|bottom&stat=<caster>` | Returns up to 20 runs as JSON. `top`/`bottom` rank by the run's optimal-land-count `score_<stat>` and require a stat from the CASTER set; invalid combinations fall back to `recent`. Each run carries a `deck_breakdown` reconstructed from the DB (`CardRow.types_json/cmc` + `DeckCardRow.quantity/is_commander`) -- commanders, mana curve, land count, ramp counts (total + per-CMC bucket), draw counts (cantrip / instant / repeatable). Legacy runs whose deck has no persisted card metadata return `deck_breakdown: null`; the row's "Show" panel exposes an "Open in config" link that re-saves the deck (POST from localStorage or GET from disk) to backfill the metadata. |

## Asset pipeline + static caching

The Pyodide wheel and the static JS/CSS files are all cache-busted via mtime
query strings so editing source on disk is immediately visible in the browser
without a hard refresh:

- **Pyodide wheel** -- `/sim/api/wheel` returns `{filename, mtime}`. The simulate page builds the download URL as `<filename>?v=<mtime>`. `pyodide_worker.js` strips the `?v=...` query string before writing the file to its in-memory FS so micropip still sees a clean `.whl`. Each `uv build --wheel` rebuild bumps the mtime, so the browser refetches.
- **Static JS / CSS** -- `web/__init__.py` registers a `static_url(filename)` Jinja helper that wraps `url_for('static', ...)` and appends `?v=<mtime>`. **Use `static_url` everywhere instead of `url_for('static', ...)`** -- otherwise edits to JS/CSS won't reach the browser without manually clearing cache.

`scripts/start_flask.sh` runs `uv build --wheel --quiet` on every start, so the wheel is always fresh after a server restart. **Editing Python source while the server is running** is auto-reloaded by Flask `--debug`, but the *Pyodide wheel* is only rebuilt on the next `start_flask.sh` invocation. Restart the script to push Python changes into the in-browser runtime.

## Adding a new top-level JS module

Both `simulate.html` (via inline `<script>`) and `deck_store.js` (via
`navigateToSim` / `navigateToManaModel`) load pages by calling
`document.write(html)`. The new HTML re-runs every `<script src=...>` tag in
`base.html`, so any global declarations in those scripts execute **twice** in
the same JS context.

`const Foo = ...` will throw `Identifier 'Foo' has already been declared` on
the second run, aborting the rest of `document.write` processing and leaving
the page non-functional. The canonical pattern is the idempotent guard used in
`client_results.js`:

```js
var Foo = window.Foo || (function() {
    'use strict';
    // ... module body ...
    return {publicMethod1, publicMethod2};
})();
if (typeof window !== 'undefined') { window.Foo = Foo; }
```

This makes the second execution a no-op rather than a fatal redeclaration.

## Configuration

- `SECRET_KEY` env var (defaults to `"dev"`)
- `DATABASE_URL` env var -- if set, enables Postgres persistence via `db/` module
- `AUTO_GOLDFISH_CALIBRATE` env var -- defaults to enabled. When the DB is reachable and contains persisted raw composite stats, the `/runs` page tunes the 1-10 score anchors against the empirical distribution (Bayesian-shrunk toward defaults). Set to `0` to fall back to built-in default anchors. The `/runs` page shows a "Calibrated" or "Default anchors" badge with the active values. The `/sim/<deck>` page injects the active anchors into `window.CASTER_CALIBRATION` so `client_results.js` can render the "What does my CASTER Score mean?" expandable panel against current calibration.
