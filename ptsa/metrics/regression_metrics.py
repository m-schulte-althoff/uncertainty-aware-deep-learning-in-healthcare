import numpy as np


from ptsa.tasks.length_of_stay.inference_probabilistic import LOSProbabilisticInference

def compute_regression_metrics(y_true, y_pred, y_variance=None):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
    }

    if y_variance is not None:
        epsilon = 1e-8
        safe_variance = np.maximum(y_variance, epsilon)

        squared_error = (y_true - y_pred) ** 2
        nll = np.mean(squared_error / (2 * safe_variance) + 0.5 * np.log(2 * np.pi * safe_variance))
        metrics["nll"] = nll

    return metrics

if __name__ == "__main__":
    MODEL = "GRU"
    PROBABILISTIC = True

    config = {
            "input_size": 38,
            "hidden_size": 95,
            "num_layers": 3,
            "learning_rate": 0.00015133860634638263,
            "dropout": 0.4790674785796453,
            "batch_size": 32,
            "num_epochs": 10,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100,
            "d_model": 128,
            "dim_feedforward": 64,
            "nhead": 8,
            }

    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/length-of-stay-fixed/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/length_of_stay/final/gru_prob_los.pth"


    inference_session = LOSProbabilisticInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name=MODEL, 
                                                  device="cuda:3",
                                                  num_batches_inference=200,
                                                  limit_num_test_sampled=True,
                                                  probabilistic=PROBABILISTIC)
    train_data , _, test_data = inference_session.load_test_data()

    predicted_means, predicted_variances, y_true = inference_session.infer_on_data_points(test_data)
    
    predicted_means = np.array(predicted_means).flatten()
    predicted_variances = np.array(predicted_variances).flatten()
    y_true = np.array(y_true).flatten()
    
    print("*" * 10)
    print(f"Model Type: {MODEL}")
    
    if PROBABILISTIC:
        metrics = compute_regression_metrics(y_true, predicted_means, predicted_variances)
    else:
        metrics = compute_regression_metrics(y_true, predicted_means)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    if PROBABILISTIC:
        print(f"UNCERTAINTY: {np.mean(predicted_variances)}")

    
