"""
ml_metrics_logger.py — Append per-trial metrics to a running CSV summary.
"""
import csv
from datetime import datetime
from sklearn.metrics import classification_report, cohen_kappa_score

def log_trial(output_dir, sensor, test_year, train_years,
              n_train, n_test, y_test, y_pred, class_names):

    target_names = [class_names[i] for i in sorted(class_names)]
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True, zero_division=0)

    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sensor":          sensor,
        "test_year":       test_year,
        "train_years":     "_".join(str(y) for y in train_years),
        "n_train":         n_train,
        "n_test":          n_test,
        "accuracy":        round(report["accuracy"],                4),
        "kappa":           round(cohen_kappa_score(y_test, y_pred), 4),
        "macro_f1":        round(report["macro avg"]["f1-score"],   4),
        "macro_precision": round(report["macro avg"]["precision"],  4),
        "macro_recall":    round(report["macro avg"]["recall"],     4),
        "weighted_f1":     round(report["weighted avg"]["f1-score"],4),
    }

    for cls_name in target_names:
        slug = cls_name.lower().replace(" ", "_")
        r    = report[cls_name]
        row[f"{slug}_precision"] = round(r["precision"], 4)
        row[f"{slug}_recall"]    = round(r["recall"],    4)
        row[f"{slug}_f1"]        = round(r["f1-score"],  4)
        row[f"{slug}_support"]   = int(r["support"])

    csv_path = output_dir / "trial_metrics.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  [metrics] Logged --> {csv_path}")