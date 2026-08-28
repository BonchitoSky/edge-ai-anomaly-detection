"""
Console output encoding.

Windows consoles default to cp1252, which cannot encode the box-drawing rules,
arrows and Greek letters these scripts use in their progress output. Printing one
raises UnicodeEncodeError and kills the run -- losing a training job to a
decorative separator. Reconfiguring stdout to UTF-8 fixes the cause; errors are
replaced rather than raised so formatting can never abort real work.

Call enable_utf8_output() before the first print in any script that reports
progress. It is idempotent and a no-op where stdout is already UTF-8.
"""

import sys


def enable_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        # Streams that are redirected to a pipe or file may not be reconfigurable
        # (io.StringIO under test, for instance); those are already fine as-is.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
