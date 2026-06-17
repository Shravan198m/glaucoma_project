import sys
import json
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_score,
                             recall_score, f1_score, accuracy_score,
                             precision_recall_curve)
from model import create_resnet50_model
from dataset import create_dataloaders, CLASS_NAMES
import os
from pathlib import Path


def predict_from_probs(probs, threshold=0.5):
    """Convert probabilities to binary labels using a decision threshold."""
    return (np.asarray(probs) >= threshold).astype(int)


def find_best_threshold(labels, probs, beta=2.0, min_precision=0.0):
    """
    Find a threshold that prioritizes recall.

    Uses the F-beta score with beta > 1 to weight recall more strongly than
    precision. beta=2 is a good default for medical screening tasks.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    if labels.size == 0:
        return 0.5, 0.0, 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    thresholds = np.append(thresholds, 1.0)

    best_threshold = 0.5
    best_score = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for idx, threshold in enumerate(thresholds):
        current_precision = precision[min(idx, len(precision) - 1)]
        current_recall = recall[min(idx, len(recall) - 1)]

        if current_precision < min_precision:
            continue

        beta_sq = beta * beta
        if current_precision + current_recall == 0:
            score = 0.0
        else:
            score = (1 + beta_sq) * current_precision * current_recall / (
                beta_sq * current_precision + current_recall
            )

        if score > best_score or (score == best_score and current_recall > best_recall):
            best_score = score
            best_threshold = float(threshold)
            best_precision = float(current_precision)
            best_recall = float(current_recall)

    return best_threshold, best_score, best_precision, best_recall


def evaluate_model(model, test_loader, device=None, threshold=0.5):
    """
    Run model on test set and collect all predictions.
    Returns true labels, predicted labels, and raw probabilities.
    """
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    model.eval()

    all_labels      = []
    all_predictions = []
    all_probs       = []

    print("Running evaluation on test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)           # Raw probabilities (0-1)
            probs   = outputs.squeeze(1).cpu().numpy()
            preds   = predict_from_probs(probs, threshold=threshold)
            lbls    = labels.numpy().astype(int)

            all_probs.extend(probs.tolist())
            all_predictions.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

        return (np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probs))


def print_metrics(labels, predictions, probs):
    """Print all classification metrics clearly."""

    acc  = accuracy_score(labels, predictions)  * 100
    prec = precision_score(labels, predictions) * 100
    rec  = recall_score(labels, predictions)    * 100
    f1   = f1_score(labels, predictions)        * 100

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    print("="*50)

    # WHY EACH METRIC MATTERS IN MEDICAL IMAGING:
    print("\nMETRIC INTERPRETATIONS:")
    print(f"  Accuracy  ({acc:.1f}%): Overall correct predictions")
    print(f"  Precision ({prec:.1f}%): Of predicted Glaucoma cases, "
          f"how many are truly Glaucoma")
    print(f"  Recall    ({rec:.1f}%): Of all actual Glaucoma cases, "
          f"how many did we catch")
    print(f"  → Recall is MOST IMPORTANT in medical screening!")
    print(f"     Missing a Glaucoma case (low recall) is worse than")
    print(f"     a false alarm (low precision)")
    print(f"  F1-Score  ({f1:.1f}%): Balance between Precision and Recall")

    print("\nDETAILED REPORT:")
    print(classification_report(labels, predictions,
                                target_names=CLASS_NAMES))

    return acc, prec, rec, f1


def plot_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0A2540")
    ax.set_facecolor("#0A2540")

    # Custom color map from Navy to Cyan
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("navy_cyan", ["#0A2540", "#0F2D4D", "#11335A", "#00C2FF"], N=256)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, color='white', fontsize=11, fontweight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, color='white', fontsize=11, fontweight='bold')

    ax.set_ylabel('Actual Label', fontsize=12, color='#B8C4D4', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, color='#B8C4D4', fontweight='bold')

    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.08))


    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha='center',
                va='center',
                color='white' if cm[i, j] > threshold else '#B8C4D4',
                fontsize=18,
                fontweight='bold',
            )

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.25, 0.85, f'TN={tn}', ha='center',
             transform=ax.transAxes, color='#10B981', fontsize=12, fontweight='bold')
    ax.text(0.75, 0.85, f'FP={fp}', ha='center',
             transform=ax.transAxes, color='#F59E0B', fontsize=12, fontweight='bold')
    ax.text(0.25, 0.15, f'FN={fn}', ha='center',
             transform=ax.transAxes, color='#EF4444', fontsize=12, fontweight='bold')
    ax.text(0.75, 0.15, f'TP={tp}', ha='center',
             transform=ax.transAxes, color='#10B981', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/confusion_matrix.png',
                dpi=150, bbox_inches='tight', facecolor="#0A2540")
    plt.close()
    print("📊 Confusion matrix saved!")


def plot_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr       = fpr[optimal_idx]
    optimal_tpr       = tpr[optimal_idx]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0A2540")
    ax.set_facecolor("#0A2540")

    ax.plot(fpr, tpr, color="#00C2FF", linewidth=3.0,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color="white", linestyle="--",
             linewidth=1.5, alpha=0.5, label='Random Classifier (AUC=0.5)')

    ax.scatter(optimal_fpr, optimal_tpr, s=180,
                color='#10B981', edgecolor='white', linewidth=2, zorder=5,
                label=f'Optimal Threshold = {optimal_threshold:.3f}\n'
                      f'(Sensitivity={optimal_tpr:.3f}, Spec.={1-optimal_fpr:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, color='#B8C4D4', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, color='#B8C4D4', fontweight='bold')

    ax.grid(True, color='white', alpha=0.06, linestyle=':', linewidth=1)
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.08))

    legend = ax.legend(loc='lower right', fontsize=11, facecolor='#0F2D4D', edgecolor=(1.0, 1.0, 1.0, 0.08))

    plt.setp(legend.get_texts(), color='white')

    plt.tight_layout()
    plt.savefig('outputs/plots/roc_curve.png',
                dpi=150, bbox_inches='tight', facecolor="#0A2540")
    plt.close()
    print(f"📊 ROC Curve saved! AUC = {roc_auc:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")

    return roc_auc, optimal_threshold


def full_evaluation(model_path='outputs/models/best_model.pth', dataset_root='dataset', batch_size=8, beta=2.0, min_precision=0.0, summary_filename='tuned_evaluation_summary.json'):
    """Run complete evaluation pipeline with recall-oriented threshold tuning."""

    device = torch.device('cpu')

    # Load best model
    print(f"Loading model from {model_path}...")
    model = create_resnet50_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    print("✅ Model loaded!")

    # Load data
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(dataset_root, batch_size=batch_size)

    # Tune threshold on validation set to favor recall
    print("Tuning decision threshold on validation set...")
    val_labels, _, val_probs = evaluate_model(model, val_loader, device=device, threshold=0.5)
    tuned_threshold, fbeta_score, tuned_precision, tuned_recall = find_best_threshold(
        val_labels,
        val_probs,
        beta=beta,
        min_precision=min_precision,
    )
    print(
        f"✅ Tuned threshold: {tuned_threshold:.4f} | "
        f"F{beta:.1f}: {fbeta_score:.4f} | Precision: {tuned_precision:.4f} | Recall: {tuned_recall:.4f}"
    )

    # Get predictions
    labels, predictions, probs = evaluate_model(model, test_loader, device=device, threshold=tuned_threshold)

    # Print all metrics
    acc, prec, rec, f1 = print_metrics(labels, predictions, probs)

    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)

    # Plot ROC curve
    roc_auc, opt_threshold = plot_roc_curve(labels, probs)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Threshold: {tuned_threshold:.4f}")
    print("="*50)

    summary_path = Path('outputs/results') / summary_filename
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "batch_size": batch_size,
        "tuning": {
            "beta": beta,
            "min_precision": min_precision,
            "threshold": float(tuned_threshold),
            "validation_fbeta": float(fbeta_score),
            "validation_precision": float(tuned_precision),
            "validation_recall": float(tuned_recall),
        },
        "test_metrics": {
            "accuracy": float(acc / 100.0),
            "precision": float(prec / 100.0),
            "recall": float(rec / 100.0),
            "f1": float(f1 / 100.0),
            "roc_auc": float(roc_auc),
            "roc_optimal_threshold": float(opt_threshold),
        },
        "confusion_matrix": {
            "TP": int(((predictions == 1) & (labels == 1)).sum()),
            "TN": int(((predictions == 0) & (labels == 0)).sum()),
            "FP": int(((predictions == 1) & (labels == 0)).sum()),
            "FN": int(((predictions == 0) & (labels == 1)).sum()),
        },
        "counts": {
            "total": int(len(labels)),
            "glaucoma_predicted": int(predictions.sum()),
            "glaucoma_actual": int(labels.sum()),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    print(f"💾 Tuned evaluation summary saved to {summary_path}")

    return labels, predictions, probs, tuned_threshold


if __name__ == "__main__":
    full_evaluation(beta=1.0, min_precision=0.85, summary_filename='tuned_balanced_summary.json')

