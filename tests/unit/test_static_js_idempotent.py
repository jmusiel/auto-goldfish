"""Regression tests for client_results.js idempotent loading.

Background: deck_store.js's ``navigateToSim`` and ``navigateToManaModel`` do
``document.write(html)`` to swap pages without a full navigation. The new HTML
re-runs every ``<script src=...>`` tag in ``base.html`` in the same JS context.

If a top-level module is declared with ``const Foo = ...``, the second load
throws ``Identifier 'Foo' has already been declared`` mid-document.write,
which aborts the rest of the document.write processing and silently leaves
the destination page non-functional (no event handlers wired up, no result
panel after a sim, etc.).

The fix in commit 3ecdef3 was to use the idempotent guard pattern
``var Foo = window.Foo || (function(){ ... })()``. These tests lock that
pattern in.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


JS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "auto_goldfish" / "web" / "static" / "js" / "client_results.js"
)


def test_client_results_uses_idempotent_declaration_pattern():
    """Source-level check: client_results.js must declare ClientResults with
    var + window guard, never const / let.
    """
    source = JS_PATH.read_text()
    assert "const ClientResults" not in source, (
        "client_results.js declares ClientResults with const, which throws on "
        "re-execution via document.write. Use `var ClientResults = "
        "window.ClientResults || (function() {...})()` instead. "
        "See commit 3ecdef3 and web/README.md."
    )
    assert "var ClientResults" in source, (
        "client_results.js must declare ClientResults with var (not let or "
        "const) so it survives re-execution via deck_store.js's document.write."
    )
    assert "window.ClientResults =" in source, (
        "client_results.js must persist ClientResults to window so the "
        "second-load guard (`window.ClientResults || (...)`) short-circuits."
    )


def test_client_results_can_be_loaded_twice_in_same_context():
    """Functional check: loading the file twice in one JS context must not
    throw. Requires node; skips when unavailable.
    """
    if not shutil.which("node"):
        pytest.skip("node not available; skipping functional JS test")

    test_script = f"""
        const fs = require('fs');
        const code = fs.readFileSync({str(JS_PATH)!r}, 'utf8');

        // Minimal DOM polyfills the IIFE touches at top level.
        const fakeEl = () => ({{
            textContent: '', innerHTML: '',
            appendChild: () => {{}}, querySelectorAll: () => [],
        }});
        global.document = {{
            createElement: fakeEl,
            getElementById: () => null,
            querySelectorAll: () => [],
            body: fakeEl(),
        }};
        global.window = {{}};
        global.Chart = class {{ static getChart() {{ return null; }} }};

        // Browser classic-script top-level `var` lives on window. Rewrite the
        // declaration so node's eval scope mimics that.
        const rewritten = code.replace(
            /^var ClientResults =/m,
            'global.ClientResults ='
        );

        // Two consecutive evals = two <script src=client_results.js> tags
        // re-running on the same window. The var-guard regression would
        // throw "Identifier 'ClientResults' has already been declared".
        eval(rewritten);
        eval(rewritten);

        if (typeof global.ClientResults !== 'object' || global.ClientResults === null) {{
            throw new Error('ClientResults was not assigned');
        }}
        if (typeof global.ClientResults.render !== 'function') {{
            throw new Error('ClientResults.render is missing after second load');
        }}
        console.log('ok');
    """

    result = subprocess.run(
        ["node", "-e", test_script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"client_results.js failed when loaded twice -- "
        f"deck_store.js's document.write path would crash too.\n"
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert "ok" in result.stdout
