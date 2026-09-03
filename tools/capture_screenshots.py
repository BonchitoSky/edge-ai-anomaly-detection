"""
Capture dashboard screenshots for slides and the README.

The demo scenario is deterministic and can be jumped to any phase, so the shots
are reproducible rather than lucky: this drives the running dashboard through the
moments worth showing and saves a PNG for each.

    python dashboard/app.py --demo          # in one terminal
    python tools/capture_screenshots.py     # in another

Each shot waits for the scenario to settle after the trigger, because the model
verdict only refreshes once per 30 s of simulated time — grabbing the frame
immediately would catch the previous verdict and misrepresent the system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    raise SystemExit(
        "Playwright is required:  pip install playwright && playwright install chromium"
    )

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"

DESKTOP = {"width": 1600, "height": 1200}
PHONE = {"width": 390, "height": 1100}

# (filename, trigger, settle seconds, viewport, full page)
SHOTS = [
    ("dashboard-baseline.png", "clear", 10, DESKTOP, True),
    ("dashboard-gas-leak.png", "gasleak", 16, DESKTOP, True),
    ("dashboard-overheat.png", "overheat", 16, DESKTOP, True),
    ("dashboard-intrusion.png", "intrusion", 16, DESKTOP, True),
    ("dashboard-mobile.png", "gasleak", 14, PHONE, False),
]


def trigger(page, event: str) -> None:
    page.evaluate(
        """(ev) => fetch('/api/simulate', {
             method: 'POST',
             headers: {'Content-Type': 'application/json'},
             body: JSON.stringify({event: ev})
           }).then(r => r.json())""",
        event,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/")
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for name, event, settle, viewport, full in SHOTS:
                ctx = browser.new_context(viewport=viewport, device_scale_factor=2)
                page = ctx.new_page()
                # Not "networkidle": the SSE stream stays open for the life of
                # the page, so the network is never idle and the wait times out.
                page.goto(args.url, wait_until="domcontentloaded")
                page.wait_for_selector(".node-card", timeout=15000)
                page.evaluate(
                    "(t) => { localStorage.setItem('edgeai.theme', JSON.stringify(t));"
                    "document.documentElement.setAttribute('data-theme', t); }",
                    args.theme,
                )
                # Let the stream fill the charts before jumping, so the history
                # behind the incident is not an empty axis.
                page.wait_for_timeout(6000)
                trigger(page, event)
                page.wait_for_timeout(settle * 1000)
                page.screenshot(path=str(OUT / name), full_page=full)
                print(f"  {name}")
                ctx.close()
        finally:
            browser.close()

    print(f"\nSaved to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
