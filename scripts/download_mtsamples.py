#!/usr/bin/env python3
"""Baixa o dataset público MTSamples (transcrições médicas)."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/mtsamples.csv"),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}")
    urlretrieve(args.url, args.output)
    print(f"saved → {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
