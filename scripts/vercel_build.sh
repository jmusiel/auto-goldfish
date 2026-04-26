#!/usr/bin/env bash
# Vercel build script: builds the Pyodide wheel and assembles static assets
# into the public/ directory for CDN serving.
set -euo pipefail

echo "==> Building wheel for Pyodide..."
pip install build
python -m build --wheel --outdir dist/

echo "==> Assembling public/ directory..."
rm -rf public
mkdir -p public/static public/dist

# Copy static assets (JS, CSS)
cp -r src/auto_goldfish/web/static/* public/static/

# Copy wheel for Pyodide download
cp dist/auto_goldfish-*.whl public/dist/

echo "==> Running database migrations (no-op when DATABASE_URL unset)..."
python scripts/migrate.py

echo "==> Build complete."
ls -la public/static/ public/dist/
