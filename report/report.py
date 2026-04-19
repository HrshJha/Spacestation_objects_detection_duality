from pathlib import Path
import shutil
import re
import csv

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
OUTPUT = ROOT / "Output"
TEST_DIR = ROOT / "testImages"

REPORT_OUTPUT = REPORT_DIR / "generatedreport.md"


def find_latest_run(prefix):
    detect_dir = OUTPUT / "runs" / "detect"
    if not detect_dir.exists():
        return None
    runs = [p for p in detect_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def parse_metrics_from_csv(run_dir):
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        return {}
    with metrics_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return {
        "images": last.get("images", "n/a"),
        "instances": last.get("instances", "n/a"),
        "precision": last.get("precision", "n/a"),
        "recall": last.get("recall", "n/a"),
        "mAP50": last.get("mAP50", "n/a"),
        "mAP50-95": last.get("mAP50-95", "n/a"),
    }


def parse_metrics_from_text(run_dir):
    text_file = run_dir / "results.txt"
    if not text_file.exists():
        return {}
    text = text_file.read_text(encoding="utf-8")
    pattern = r"all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
    match = re.search(pattern, text)
    if not match:
        return {}
    return {
        "images": match.group(1),
        "instances": match.group(2),
        "precision": match.group(3),
        "recall": match.group(4),
        "mAP50": match.group(5),
        "mAP50-95": match.group(6),
    }


def parse_metrics(run_dir):
    if run_dir is None:
        return {}
    metrics = parse_metrics_from_csv(run_dir)
    return metrics if metrics else parse_metrics_from_text(run_dir)


def copy_file(src, dest_name):
    if src.exists():
        dest = REPORT_DIR / dest_name
        shutil.copy(src, dest)
        return dest.name
    return None


def sample_images(limit=6):
    predictions_images = OUTPUT / "predictions" / "images"
    if not predictions_images.exists():
        return []
    images = sorted(predictions_images.glob("*.png")) + sorted(predictions_images.glob("*.jpg"))
    copied = []
    for i, img in enumerate(images[:limit], start=1):
        dest = REPORT_DIR / f"prediction_{i}{img.suffix}"
        shutil.copy(img, dest)
        copied.append(dest.name)
    return copied


def generate_confusion_matrix():
    true_dir = TEST_DIR / "labels"
    pred_dir = OUTPUT / "predictions" / "labels"
    if not true_dir.exists() or not pred_dir.exists():
        return None
    true_labels = []
    pred_labels = []
    for true_file in sorted(true_dir.glob("*.txt")):
        with true_file.open("r", encoding="utf-8") as f:
            true_lines = [line.strip() for line in f if line.strip()]
        true_labels.append(int(true_lines[0].split()[0]) if true_lines else -1)
        pred_file = pred_dir / true_file.name
        if pred_file.exists():
            with pred_file.open("r", encoding="utf-8") as pf:
                pred_lines = [line.strip() for line in pf if line.strip()]
            pred_labels.append(int(pred_lines[0].split()[0]) if pred_lines else -1)
        else:
            pred_labels.append(-1)
    if len(true_labels) != len(pred_labels) or not true_labels:
        return None
    labels = sorted({l for l in true_labels + pred_labels if l >= 0})
    if not labels:
        return None
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(l) for l in labels])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    path = REPORT_DIR / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path.name


def build_report(train_metrics, val_metrics, train_image, val_image, sample_images, confusion_image, classes):
    lines = [
        "# YOLOv8 Synthetic Data Training Report",
        "",
        "## Summary",
        "This report summarizes the YOLOv8 synthetic-data training workflow, the real-world evaluation process, and benchmark metrics.",
        "",
        "## Dataset",
        "- Train: `Output/train/images` and `Output/train/labels`",
        "- Validation: `Output/val/images` and `Output/val/labels`",
        "- Test: `testImages/images` and `testImages/labels`",
        f"- Classes: {', '.join(classes) if classes else 'unknown'}",
        "",
        "## Metrics",
        "| Stage | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|---|---|---|---|",
        "| Training | {} | {} | {} | {} | {} | {} |".format(
            train_metrics.get("images", "n/a"),
            train_metrics.get("instances", "n/a"),
            train_metrics.get("precision", "n/a"),
            train_metrics.get("recall", "n/a"),
            train_metrics.get("mAP50", "n/a"),
            train_metrics.get("mAP50-95", "n/a"),
        ),
        "| Validation | {} | {} | {} | {} | {} | {} |".format(
            val_metrics.get("images", "n/a"),
            val_metrics.get("instances", "n/a"),
            val_metrics.get("precision", "n/a"),
            val_metrics.get("recall", "n/a"),
            val_metrics.get("mAP50", "n/a"),
            val_metrics.get("mAP50-95", "n/a"),
        ),
        "",
        "## Visual artifacts",
    ]
    if train_image:
        lines.append(f"- Training graph: `{train_image}`")
    if val_image:
        lines.append(f"- Validation result image: `{val_image}`")
    if sample_images:
        lines.append("- Prediction examples:")
        for image in sample_images:
            lines.append(f"  - `{image}`")
    if confusion_image:
        lines.append(f"- Confusion matrix: `{confusion_image}`")
    lines.extend([
        "",
        "## Observations",
        "- Training and validation metrics should be reviewed for overfitting and performance consistency.",
        "- The confusion matrix highlights class confusion and missed labels.",
        "- Prediction examples show correct and incorrect detections for visual analysis.",
        "",
        "## Next steps",
        "1. Review the graphs to confirm stable loss and improving mAP.",
        "2. Investigate any class with low recall or precision.",
        "3. Add synthetic examples for difficult object poses or backgrounds.",
        "4. Re-run training and generate a new report.",
    ])
    return "\n".join(lines)


def main():
    train_run = find_latest_run("train")
    val_run = find_latest_run("val")
    train_metrics = parse_metrics(train_run)
    val_metrics = parse_metrics(val_run)

    train_image = None
    val_image = None
    if train_run:
        train_candidate = train_run / "results.png"
        if train_candidate.exists():
            train_image = copy_file(train_candidate, "train_results.png")
    if val_run:
        val_candidate = val_run / "val_batch2_pred.jpg"
        if not val_candidate.exists():
            candidates = list(val_run.glob("*.png")) + list(val_run.glob("*.jpg"))
            val_candidate = candidates[0] if candidates else None
        if val_candidate and val_candidate.exists():
            val_image = copy_file(val_candidate, "val_results.png")

    sample_files = sample_images(limit=6)
    confusion_image = generate_confusion_matrix()
    classes = []
    class_file = OUTPUT / "classes.txt"
    if class_file.exists():
        classes = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    report_text = build_report(train_metrics, val_metrics, train_image, val_image, sample_files, confusion_image, classes)
    REPORT_OUTPUT.write_text(report_text, encoding="utf-8")
    print(f"Generated report: {REPORT_OUTPUT}")
    if train_image:
        print(f"Copied training graph: {train_image}")
    if val_image:
        print(f"Copied validation image: {val_image}")
    if sample_files:
        print(f"Copied sample prediction images: {', '.join(sample_files)}")
    if confusion_image:
        print(f"Saved confusion matrix: {confusion_image}")


if __name__ == "__main__":
    main()
