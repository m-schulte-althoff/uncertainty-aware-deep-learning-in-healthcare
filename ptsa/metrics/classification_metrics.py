import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, log_loss, brier_score_loss, auc
)
import matplotlib.pyplot as plt

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float = 0.5,
    uncertainties: np.ndarray = None
    ):

    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    brier = brier_score_loss(y_true, y_pred_proba)
    log_loss_value = log_loss(y_true, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    prevalence = np.mean(y_true)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,  # Same as recall
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "brier_score": brier,
        "log_loss": log_loss_value,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "prevalence": prevalence,
        "negative_predictive_value": npv
    }

    if uncertainties:
        epsilon = 1e-15
        p_y = np.copy(y_pred_proba)
        p_y = np.clip(p_y, epsilon, 1 - epsilon)
        p_y_true = np.where(y_true == 1, p_y, 1 - p_y)
        nll = -np.mean(np.log(p_y_true))

        metrics["nll"] = nll

    return metrics

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve_classification.png")


def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')

    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline (prevalence = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig("precision_recall_curve_classification.png")
    
    return avg_precision

if __name__ == "__main__":
    PROBABILISTIC_MODEL = True
    model_type = "RNN"

    config = {
            "input_size": 38,
            "hidden_size": 206,
            "num_layers": 2,
            "learning_rate": 0.00034262704911951214,
            "dropout": 0.5687026422892869,
            "batch_size": 64,
            "num_epochs": 26,
            "weight_decay": 0.0010025992862318614,
            "num_mc_samples": 100,
            "d_model": 64,
            "dim_feedforward": 320,
            "nhead": 4
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality-own-final"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/final/final_gru_prob_ihm.pth"


    inference_session = IHMModelInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name=model_type, 
                                                  device="cuda:3",
                                                  probabilistic=PROBABILISTIC_MODEL
                                                  )
    train_data, _, test_data = inference_session.load_test_data()

    predictions, y_true, all_uncertainties = inference_session.infer_on_data_points(test_data)
    
    predictions = np.array(predictions).flatten()
    all_uncertainties = np.array(all_uncertainties).flatten()
    if PROBABILISTIC_MODEL:
        print(f"UNCERTAINTY: {np.mean(all_uncertainties)}")
    y_true = np.array(y_true).flatten()

    if PROBABILISTIC_MODEL:
        classification_metrics = compute_classification_metrics(y_true, predictions, uncertainties=all_uncertainties)
    else:
        classification_metrics = compute_classification_metrics(y_true, predictions)
    print("*" * 20)
    print(f"Model Type: {model_type}")
    print(classification_metrics)

    plot_roc_curve(y_true, predictions)

    print("*" * 20)
    avg_precision = plot_precision_recall_curve(y_true, predictions)
    print(f"AVG Precision: {avg_precision}")

