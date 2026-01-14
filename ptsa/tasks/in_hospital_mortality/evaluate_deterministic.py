import argparse
import json
import os
import re
from typing import Dict, Any

import numpy as np
import torch

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference
from ptsa.metrics.classification_metrics import compute_classification_metrics
from ptsa.metrics.expected_calibration_error import compute_ece


def parse_best_hyperparams_txt(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()

            if re.fullmatch(r"[-+]?\d+", v):
                out[k] = int(v)
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _json_default(o: Any):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main():
    p = argparse.ArgumentParser()

    # Required I/O
    p.add_argument("--data", type=str, required=True, help="Dataset root (must contain train/ and test/)")
    p.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")

    # Architecture (direct or via best_hparams_file)
    p.add_argument("--best_hparams_file", type=str, default=None)
    p.add_argument("--input_size", type=int, default=38)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.2)

    # Metrics
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ece_bins", type=int, default=10)

    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    if (args.hidden_size is None or args.num_layers is None) and args.best_hparams_file:
        hp = parse_best_hyperparams_txt(args.best_hparams_file)
        args.hidden_size = args.hidden_size if args.hidden_size is not None else int(hp["hidden_size"])
        args.num_layers = args.num_layers if args.num_layers is not None else int(hp["num_layers"])

    if args.hidden_size is None or args.num_layers is None:
        raise SystemExit("Missing --hidden_size and/or --num_layers (pass them or use --best_hparams_file).")

    config = {
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_mc_samples": 1,
    }

    inference = IHMModelInference(
        config=config,
        data_path=args.data,
        model_path=args.model_path,
        model_name="LSTM",
        device=args.device,
        probabilistic=False,
    )

    _, _, test_data = inference.load_test_data()
    preds, y_true, _ = inference.infer_on_data_points(test_data)

    y_pred = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred_proba=y_pred,
        threshold=args.threshold,
        uncertainties=None,
    )
    ece, _, _, _ = compute_ece(y_true, y_pred, n_bins=args.ece_bins)
    metrics["ece"] = float(ece)

    metrics = to_jsonable(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    print(json.dumps(metrics, indent=2, default=_json_default))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()