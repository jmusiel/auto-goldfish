"""One-shot schema migration for auto_goldfish.

Reads ``DATABASE_URL`` from the environment and runs the same
``create_all`` + ``_migrate`` logic that previously fired on every cold
serverless worker. Intended to be invoked from the Vercel build step (or
manually before deploy) so the runtime can boot with
``AUTO_GOLDFISH_SKIP_MIGRATE=1`` and skip the per-cold-start schema work.

Idempotent and safe to run repeatedly. If ``DATABASE_URL`` is empty, exits
cleanly so the build step can run unchanged in environments without a DB.

Usage:

    DATABASE_URL=postgresql://... python scripts/migrate.py
"""
from __future__ import annotations

import logging
import os
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("auto_goldfish.migrate")


def main() -> int:
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        logger.warning("DATABASE_URL not set; skipping migration")
        return 0

    # Imported here so the script can be invoked from environments that
    # don't have the package installed yet (the importer just exits).
    from auto_goldfish.db.session import init_db

    # Force migrations on regardless of AUTO_GOLDFISH_SKIP_MIGRATE: the
    # build step is exactly when we want them to run.
    init_db(db_url, run_migrations=True)
    logger.info("Migration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
