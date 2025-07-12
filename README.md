# Probabilistic Time Series Analysis (PTSA)

This project contains several deterministic and probabilistic Neural Networks to classify In-hospital Mortality and predict the Length-of-Stay of ICU patients.

## Features

- **Probabilistic Models**: Support for various probabilistic time series models.
- **Visualizations**: Generate calibration plots and other visualizations.
- **Customizable Configurations**: Flexible model and training parameters.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lsch0lz/uncertainty-aware-deep-learning-in-healthcare.git
   cd uncertainty-aware-deep-learning-in-healthcare
   pip install -e .
   ```
   
2. Train networks using the following command:

Classification:
```bash
python ptsa/tasks/in_hospital_mortality/train_deterministic.py --network ptsa/models/deterministic/lstm.py --partition custom --data PATH_TO_YOUR_DATA --model lstm  --model_name model_name.pth --output_dir PATH_TO_OUTPUT_FOLDER
```

Regression:
```bash
python ptsa/tasks/length_of_stay/optimize_probabilistic.py --network ptsa/models/probabilistic/rnn.py --partition custom --data PATH_TO_YOUR_DATA --model rnn  --model_name model_name.pth --output_dir PATH_TO_OUTPUT_FOLDER --num_mc_samples 25 --num_trials 10 --dataset_fraction 0.1
```

3. Visualise uncertainty plots
Parameterise the visualizations inside the corresponding files.

Classification (Expected Calibration Error):
```bash
python ptsa/metrics/expected_calibration_error.py
```

Regression (Error-based Calibration):
```bash
python ptsa/metrics/calibration_error.py
```