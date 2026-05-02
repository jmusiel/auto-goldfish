"""Flask web application for auto_goldfish simulations."""

from __future__ import annotations

import os

from flask import Flask, url_for


def create_app() -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "auto-goldfish-dev-key")

    # Cache-bust static assets by appending the file's mtime as a query string.
    # Browsers cache each unique URL (e.g. `?v=1745` vs `?v=2000`) separately,
    # so editing a static file on disk forces a fresh download on the next
    # page load -- but unchanged files still hit cache normally. Templates
    # should call `static_url('js/foo.js')` instead of
    # `url_for('static', filename='js/foo.js')`; otherwise stale JS/CSS will
    # be served from browser cache after edits, which is hard to diagnose.
    @app.context_processor
    def _inject_static_url():
        def static_url(filename: str) -> str:
            """Cache-busted static URL.

            Use everywhere `url_for('static', filename=...)` would otherwise
            be called. Falls back gracefully (no `?v=` suffix) if the file
            can't be stat'd.
            """
            url = url_for("static", filename=filename)
            try:
                full_path = os.path.join(app.static_folder or "", filename)
                mtime = int(os.path.getmtime(full_path))
                sep = "&" if "?" in url else "?"
                return f"{url}{sep}v={mtime}"
            except OSError:
                return url

        return {"static_url": static_url}

    from .routes import register_blueprints

    register_blueprints(app)

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        from auto_goldfish.db.session import init_db

        init_db(db_url)

    return app
