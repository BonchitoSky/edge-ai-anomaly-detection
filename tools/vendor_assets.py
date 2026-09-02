"""
Re-download the dashboard's vendored front-end assets.

The dashboard must render with no internet: the entire project claims fully
offline operation, and a venue with no WiFi would otherwise blank the page. So
Chart.js and the two font families are committed under
dashboard/static/vendor/ rather than pulled from a CDN at runtime.

Run this only when a version needs bumping:

    python tools/vendor_assets.py
"""

import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = ROOT / "dashboard" / "static" / "vendor"

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Both families are variable fonts, so one file per family covers every weight.
ASSETS = {
    "chart.umd.min.js": "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js",
    "fonts/Inter-latin.woff2": (
        "https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2"
    ),
    "fonts/JetBrainsMono-latin.woff2": (
        "https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbv2o-flEEny0FZhsfKu5WU4zr3E_BX0"
        "PnT8RD8yKwBNntkaToggR7BYRbKPxDcwg.woff2"
    ),
}


def main() -> int:
    for rel, url in ASSETS.items():
        dest = VENDOR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        dest.write_bytes(data)
        print(f"  {len(data) / 1024:7.1f} KB  {rel}")
    print("\nfonts.css is hand-maintained - update it only if a family changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
