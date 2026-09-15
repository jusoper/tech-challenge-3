#!/usr/bin/env python3
"""CLI: executa ingest → train → save do classificador de laudos."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triagem.pipeline.run import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_training_pipeline(raw_csv=args.raw_csv, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
