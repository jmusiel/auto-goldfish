# auto_goldfish

A Hyperband-scheduled Monte Carlo global optimizer for goldfishing commander games.

Runs "goldfishing" simulations (playing games without an opponent) to evaluate deck performance, mana curves, and consistency metrics across thousands of games.

## Features

- **Goldfishing engine** -- simulates drawing, mulligans, land drops, and spell casting over N turns
- **Parallel simulation** -- uses multiple CPU cores via `ProcessPoolExecutor` for fast results (configurable `workers` parameter, defaults to all CPUs)
- **Data-driven card effects** -- ~4,750 cards with special abilities (ramp, draw, cost reduction) defined as composable effects in `card_effects.json`. 109 hand-curated + ~4,640 auto-labeled via LLM (Gemini/Ollama/Claude)
- **Archidekt integration** -- pull decklists directly from Archidekt URLs via the API
- **Draw/ramp optimization** -- uses Hyperband to enumerate generic draw and ramp card additions across land counts, finding optimal configurations. Set max draw/ramp additions to 0 for a simple land-only sweep
- **Card performance analysis** -- identifies which cards are overrepresented in high- vs low-performing games
- **Game replay viewer** -- interactive turn-by-turn replay of sample games from top/mid/low quartiles, showing hand state, played cards, board state, and mana production (works in both sequential and parallel modes)
- **Web UI** -- Flask-based dashboard for importing decks, running simulations, and viewing inline results with charts and replay viewer. Card effects editor lets you override effects before running, with overrides persisted across sessions. Results appear inline below the form for an iterative tweak-and-rerun workflow
- **Client-side simulation** -- simulations run entirely in-browser via Pyodide (CPython compiled to WebAssembly). The Flask server is a thin data layer; all compute happens on the user's hardware with a progress bar and full results rendering
- **Reports** -- generates text reports with per-bucket game stats and mana curve scatter plots (PNG)

## Setup

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev
```

Or install in editable mode:

```bash
uv pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Run with a saved deck JSON
.venv/bin/python -m auto_goldfish.cli.main --deck_name vren --deck_url https://archidekt.com/decks/19226307/vrens_murine_marauders

# Sweep land counts 36-39, 8 turns, 10k sims
.venv/bin/python -m auto_goldfish.cli.main --deck_name vren --min_lands 36 --max_lands 39 --turns 8 --sims 10000

# See all options
.venv/bin/python -m auto_goldfish.cli.main --help
```

### Web UI

```bash
# Start the Flask development server
.venv/bin/flask --app src.auto_goldfish.web:create_app run --debug
```

Then open http://127.0.0.1:5000 to import decks, run simulations, and explore results including the interactive game replay viewer.

Simulations run client-side via Pyodide (WebAssembly) -- the Flask server serves deck data and the UI, but all simulation compute happens in the browser. The first run takes ~10s to load the engine; subsequent runs are fast. Build the wheel first with `uv build --wheel` so the endpoint can serve it.

### Deploying to Vercel

The app deploys to Vercel as a serverless Flask function with static assets served from the CDN. Simulations run client-side via Pyodide — Vercel only handles the thin data layer (deck imports, DB persistence, template rendering).

1. Install the [Vercel CLI](https://vercel.com/docs/cli) or connect your GitHub repo in the Vercel dashboard
2. Set environment variables in Vercel project settings:
   - `DATABASE_URL` (optional) — Neon Postgres connection string
   - `SECRET_KEY` — Flask secret key for sessions
3. Deploy:

```bash
vercel
```

The build script (`scripts/vercel_build.sh`) automatically builds the Pyodide wheel and copies static assets to `public/` for CDN serving. The `vercel.json` config routes static files directly and everything else to the Flask serverless function.

### Database Persistence (optional)

Simulation results and deck card labels can be persisted to a Neon Postgres database. Install the `db` extra and set `DATABASE_URL`:

```bash
uv sync --extra db
DATABASE_URL="postgresql://user:pass@host/dbname" .venv/bin/flask --app src.auto_goldfish.web:create_app run
```

Tables are created automatically on first startup. If `DATABASE_URL` is not set, the app works without a database (all data in-memory/disk as before).

### As a library

```python
from auto_goldfish.decklist.loader import load_decklist
from auto_goldfish.engine.goldfisher import Goldfisher

deck = load_decklist("vren")
gf = Goldfisher(deck, turns=8, sims=1000)
result = gf.simulate()

print(f"Mean mana spent: {result.mean_mana:.1f}")
print(f"Consistency: {result.consistency:.3f}")
print(f"Bad turns: {result.mean_bad_turns:.2f}")
```

### Adding a new card

All card effects live in `src/auto_goldfish/effects/card_database.py`. No subclasses needed:

```python
# Single mana producer
reg.register("My New Rock", CardEffects(on_play=[ProduceMana(1)], ramp=True))

# Multi-effect card (draws on creature cast + produces mana)
reg.register("My Engine", CardEffects(
    cast_trigger=[PerCastDraw(creature=1)],
    on_play=[ProduceMana(2)],
    priority=1,
))
```

## Project Structure

```
src/auto_goldfish/
├── models/          # Card dataclass, GameState dataclass
├── effects/         # Effect protocols, registry, builtin effects, card database
├── engine/          # Goldfisher simulation, mana calculation, mulligan strategy
├── metrics/         # MetricsCollector, built-in metrics, aggregation, reporting
├── decklist/        # JSON loader, Archidekt API, deck builder
├── autocard/        # LLM-powered card effect labeling pipeline (Gemini/Ollama/Claude)
├── db/              # Optional Neon Postgres persistence (SQLAlchemy 2.0)
├── web/             # Flask web UI (routes, templates, simulation runner)
│   └── static/js/   # Client-side JS (Pyodide worker, results renderer)
├── pyodide_runner.py # Entry point for client-side Pyodide simulations
└── cli/             # CLI entry point

tests/
├── unit/            # Unit tests covering all modules
└── integration/     # Goldfisher end-to-end tests
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--deck_name` | `vren` | Name used for saving/loading deck JSON |
| `--deck_url` | — | Archidekt deck URL (fetches and caches) |
| `--turns` | `8` | Turns per simulated game |
| `--sims` | `10000` | Number of games to simulate |
| `--min_lands` | `36` | Start of land count sweep |
| `--max_lands` | `39` | End of land count sweep |
| `--cuts` | — | Card names to cut when adding lands |
| `--record_results` | `quartile` | Recording granularity (`centile`, `decile`, `quartile`) |
| `--verbose` | off | Print every game log |
