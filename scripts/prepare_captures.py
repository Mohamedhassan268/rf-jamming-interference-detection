"""Validate a captured/shifted RF session without inventing metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.captures import load_capture_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    try:
        session = load_capture_session(args.session)
    except (FileNotFoundError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Validated {len(session.samples)} windows with shape {session.samples.shape}.")
    print(f"Metadata fields preserved: {sorted(session.metadata)}")


if __name__ == "__main__":
    main()
