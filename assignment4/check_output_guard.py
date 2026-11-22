"""Offline check of a literal output filter using a synthetic marker."""
import json
import logging

LOGGER = logging.getLogger("output_guard")
BLOCKED = "[output blocked]"


def filter_output(text: str, protected_marker: str) -> str:
    if not protected_marker:
        raise ValueError("A protected marker must be configured.")
    if protected_marker in text:
        LOGGER.warning("Blocked an output containing a protected marker.")
        return BLOCKED
    return text


def run_checks() -> dict:
    marker = "DEMO_ONLY_MARKER_41"
    cases = [
        ("benign output", "Here is general help.", "Here is general help."),
        ("partial match", "DEMO_ONLY_MARKER", "DEMO_ONLY_MARKER"),
        ("standalone marker", marker, BLOCKED),
        ("embedded marker", f"Reply: {marker}.", BLOCKED),
    ]
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    LOGGER.addHandler(handler)
    try:
        outcomes = {
            name: filter_output(sample, marker) == expected
            for name, sample, expected in cases
        }
    finally:
        LOGGER.removeHandler(handler)
    outcomes["blocked events omit marker"] = (
        len(records) == 2 and all(marker not in message for message in records)
    )
    try:
        filter_output("Routine reply.", "")
    except ValueError:
        outcomes["empty configuration rejected"] = True
    else:
        outcomes["empty configuration rejected"] = False
    return outcomes


if __name__ == "__main__":
    results = run_checks()
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(results.values()) else 1)
