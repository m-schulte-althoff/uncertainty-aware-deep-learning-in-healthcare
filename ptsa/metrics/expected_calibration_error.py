import numpy as np

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference

def compute_ece(y_true, y_pred, n_bins=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(len(y_pred)):
        bin_sums[binids[i]] += y_pred[i]
        bin_true[binids[i]] += y_true[i]
        bin_counts[binids[i]] += 1

    bin_confidences = bin_sums / (bin_counts + 1e-8)
    bin_accuracies = bin_true / (bin_counts + 1e-8)

    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * (bin_counts / len(y_pred)))
    
    return ece, bin_confidences, bin_accuracies, bin_counts


if __name__ == "__main__":
    PROBABILISTIC_MODEL = True

    config = {
            "input_size": 38,
            "hidden_size": 117,
            "num_layers": 4,
            "learning_rate": 0.00003436789697193355,
            "dropout": 0.2908014307312093,
            "batch_size": 32,
            "num_epochs": 36,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100,
            "d_model": 64,
            "dim_feedforward": 320,
            "nhead": 4
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality-own-final"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/final/final_rnn_prob_ihm.pth"


    inference_session = IHMModelInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name="RNN", 
                                                  device="cuda:0",
                                                  probabilistic=PROBABILISTIC_MODEL
                                                  )
    train_data, _, test_data = inference_session.load_test_data()

    predictions, y_true, all_uncertainties = inference_session.infer_on_data_points(test_data)
    
    predictions = np.array(predictions).flatten()
    all_uncertainties = np.array(all_uncertainties).flatten()
    y_true = np.array(y_true).flatten()

    ece, bin_confidence, bin_acc, bin_counts = compute_ece(y_true, predictions)

    ece, confidences, accuracies, counts = compute_ece(y_true, predictions)
    print(f"Expected Calibration Error: {ece:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Identity Line")
    
    plt.plot(confidences, accuracies, 'b-', linewidth=2, label='Model calibration')
    plt.fill_between(confidences, 
                    accuracies, 
                    confidences,
                    alpha=0.2,
                    color='blue',
                    label='Miscalibration area')
    plt.plot(confidences, accuracies, 'b.', markersize=10)

    plt.text(0.05, 0.95, f'Expected Calibration Error = {ece:.2f}', 
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.title("Calibration Curve")

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.savefig("ece_plot.png")



