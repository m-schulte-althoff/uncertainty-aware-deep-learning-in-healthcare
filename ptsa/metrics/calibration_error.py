import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MaxNLocator

from ptsa.tasks.length_of_stay.inference_probabilistic import LOSProbabilisticInference


def calculate_bca_confidence_intervals(data, statistic_func, confidence_level=0.95, n_bootstraps=1000):
    """
    Calculate BCa (bias-corrected and accelerated) bootstrap confidence intervals
    See: Levi et al. (Evaluating and Calibrating Uncertainty Prediction in  Regression Tasks)
    See: Rasmussen et al. (Uncertain of uncertainties? A comparison of uncertainty quantification metrics for chemical data sets)
    """
    n = len(data)
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))

    # Calculate bias correction factor
    observed_stat = statistic_func(data)
    prop_less = np.mean(np.array(bootstrap_stats) < observed_stat)
    z0 = stats.norm.ppf(prop_less)

    # Calculate acceleration factor
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jackknife_sample))
    jackknife_mean = np.mean(jackknife_stats)
    num = np.sum((jackknife_mean - jackknife_stats) ** 3)
    den = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5
    a = num / (den + np.finfo(float).eps)  # Add small epsilon to prevent division by zero

    # Calculate BCa intervals
    z_alpha = stats.norm.ppf((1 - confidence_level) / 2)
    z_1_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    for z in [z_alpha, z_1_alpha]:
        w = z0 + (z0 + z) / (1 - a * (z0 + z))
        p = stats.norm.cdf(w)
        yield np.percentile(bootstrap_stats, p * 100)

def create_error_based_calibration_plot(y_true, predicted_means, predicted_variances, n_bins=20):
    errors = abs(y_true - predicted_means)

    sorted_indices = np.argsort(predicted_variances)
    sorted_errors = errors[sorted_indices]
    sorted_variances = predicted_variances[sorted_indices]

    bin_size = len(y_true) // n_bins
    
    rmse_values = []
    rmv_values = []
    ci_lower = []
    ci_upper = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)

        bin_errors = sorted_errors[start_idx:end_idx]
        bin_variances = sorted_variances[start_idx:end_idx]
        
        rmse = np.sqrt(np.mean(bin_errors**2))
        rmv = np.sqrt(np.mean(bin_variances))
        
        rmse_values.append(rmse)
        rmv_values.append(rmv)
        
        ci = list(calculate_bca_confidence_intervals(
            bin_errors,
            lambda x: np.sqrt(np.mean(x**2)),
            confidence_level=0.95
        ))
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    rmse_values = np.array(rmse_values)
    rmv_values = np.array(rmv_values)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    slope, intercept = np.polyfit(rmv_values, rmse_values, 1)
    r_squared = np.corrcoef(rmv_values, rmse_values)[0,1]**2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(rmv_values, ci_lower, ci_upper, alpha=0.2, color='blue',
                   label='95% Confidence Interval')

    ax.scatter(rmv_values, rmse_values, color='blue')

    x_max = np.max(rmv_values) * 1.2  # 20% padding
    y_max = max(np.max(rmse_values), x_max) * 1.2
    
    max_limit = max(np.max(rmv_values), np.max(rmse_values)) * 1.2

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    x_range = np.array([0, x_max])
    ax.plot(x_range, slope * x_range + intercept, 'r--', 
             label=f'R²={r_squared:.2f}')

    ax.plot([0, max_limit], [0, max_limit], color="k", linestyle="--", linewidth=1, label='Identity Line')

    ax.set_xlabel('RMV (Root Mean Variance)', fontsize=12, labelpad=10)
    ax.set_ylabel('RMSE (Root Mean Square Error)', fontsize=12, labelpad=10)
    ax.set_title('Error-based Calibration Plot', fontsize=14, pad=20)
    ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_aspect('equal', adjustable='box')

    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    
    metrics = {
        'R2': r_squared,
        'slope': slope,
        'intercept': intercept
    }
    
    return fig, metrics

if __name__ == "__main__":
    config = {
            "input_size": 47,
            "hidden_size": 1,
            "num_layers": 1,
            "learning_rate": 0.00015133860634638263,
            "dropout": 0.4372281049790314,
            "batch_size": 64,
            "num_epochs": 10,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100,
            "d_model": 128,
            "dim_feedforward": 384,
            "nhead": 2,
            }

    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/length-of-stay-own/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/length_of_stay/probabilistic/final/transformer_final_model.pth"


    inference_session = LOSProbabilisticInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name="transformer", 
                                                  device="cuda:3",
                                                  num_batches_inference=200,
                                                  limit_num_test_sampled=True)
    train_data , _, test_data = inference_session.load_test_data()

    predicted_means, predicted_variances, y_true = inference_session.infer_on_data_points(test_data)
    
    predicted_means = np.array(predicted_means).flatten()
    predicted_variances = np.array(predicted_variances).flatten()
    y_true = np.array(y_true).flatten()
    
    sort_idx = np.argsort(predicted_variances)
    y_true = y_true[sort_idx]
    predicted_means = predicted_means[sort_idx]
    predicted_variances = predicted_variances[sort_idx]

    fig, metrics = create_error_based_calibration_plot(
        y_true, predicted_means, predicted_variances, n_bins=20
    )
    
    print("Calibration Metrics:")
    print(f"R-Squared = {metrics['R2']:.3f}")
    print(f"Slope = {metrics['slope']:.3f}")
    print(f"Intercept = {metrics['intercept']:.3f}")
   
    fig.savefig('calibration_plot_hq.png', dpi=300)
