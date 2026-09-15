#!/usr/bin/env python3
"""Exporta o joblib sklearn existente para ONNX (sem retreinar)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from triagem.pipeline.export_onnx import export_pipeline_to_onnx
from triagem.pipeline.paths import model_artifact_path, onnx_artifact_path, onnx_labels_path
from triagem.pipeline.train import load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joblib", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = load_pipeline(args.joblib or model_artifact_path())
    onnx_path, labels_path = export_pipeline_to_onnx(
        pipeline,
        args.onnx or onnx_artifact_path(),
        labels_path=onnx_labels_path(),
    )
    print(json.dumps({"onnx_path": str(onnx_path), "labels_path": str(labels_path)}, indent=2))


if __name__ == "__main__":
    main()
