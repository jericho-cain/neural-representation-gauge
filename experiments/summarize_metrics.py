import json
from pathlib import Path


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    out_md = artifacts / "metrics_table.md"

    metrics_files = [
        artifacts / "metrics_digits_nn.json",
        artifacts / "metrics_cifar10_nn.json",
    ]
    rows = [m for m in (load_metrics(p) for p in metrics_files) if m is not None]

    header = "| setting | mean(|Δcos|) | Jaccard@10 | max |Δlogits| | pred agreement |\n"
    sep = "|---|---:|---:|---:|---:|\n"
    body = []
    for m in rows:
        j10 = m.get("mean_jaccard_at_k", {}).get("10", float("nan"))
        body.append(
            "| {setting} | {delta:.4f} | {j10:.4f} | {logit:.2e} | {agree:.4f} |".format(
                setting=m.get("setting", "unknown"),
                delta=float(m.get("mean_abs_delta_cos", float("nan"))),
                j10=float(j10),
                logit=float(m.get("max_abs_logit_diff", float("nan"))),
                agree=float(m.get("prediction_agreement", float("nan")),),
            )
        )

    out_text = "# Gauge Distortion Summary\n\n"
    if not body:
        out_text += "No metrics JSON files found. Run experiment scripts first.\n"
    else:
        out_text += header + sep + "\n".join(body) + "\n"

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(out_text)
    print(f"Saved summary table: {out_md}")


if __name__ == "__main__":
    main()
