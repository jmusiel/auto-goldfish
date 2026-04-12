"""Blueprint registration."""

from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from .dashboard import bp as dashboard_bp
    from .decks import bp as decks_bp
    from .mana_model import bp as mana_model_bp
    from .runs import bp as runs_bp
    from .simulation import bp as simulation_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(decks_bp)
    app.register_blueprint(runs_bp)
    app.register_blueprint(simulation_bp)
    app.register_blueprint(mana_model_bp)
