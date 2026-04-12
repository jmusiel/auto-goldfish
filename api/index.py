"""Vercel serverless entry point — exposes the Flask app as a WSGI handler."""

from auto_goldfish.web import create_app

app = create_app()
