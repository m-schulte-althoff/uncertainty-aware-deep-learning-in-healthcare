"""
Pytest test cases for deterministic and probabilistic training scripts.

Tests cover:
- Both deterministic and probabilistic training modes
- All model architectures: LSTM, RNN, GRU, Transformer
- Both datasets: MIMIC (in-hospital-mortality-fixed) and INALO (in-hospital-mortality-own-final)
- Uses --small_part flag to limit data samples for faster testing
- Uses --num_trials 1 for probabilistic tests to minimize runtime

Run with: pytest tests/test_training.py -v
Or for a specific test: pytest tests/test_training.py::test_deterministic_lstm_mimic -v
"""

import os
import sys
import subprocess
import tempfile
import shutil
import pytest

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")

# Check if venv exists, otherwise use system python
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# Datasets
DATASETS = {
    "MIMIC": os.path.join(PROJECT_ROOT, "data", "in-hospital-mortality-fixed"),
    "INALO": os.path.join(PROJECT_ROOT, "data", "in-hospital-mortality-own-final"),
}

# Models
MODELS = ["lstm", "rnn", "gru", "transformer"]

# Training scripts
DETERMINISTIC_SCRIPT = os.path.join(
    PROJECT_ROOT, "ptsa", "tasks", "in_hospital_mortality", "train_deterministic.py"
)
PROBABILISTIC_SCRIPT = os.path.join(
    PROJECT_ROOT, "ptsa", "tasks", "in_hospital_mortality", "train_probabilistic.py"
)


def run_training_command(cmd, timeout=600):
    """
    Run a training command and return the result.
    
    Args:
        cmd: List of command arguments
        timeout: Maximum time in seconds to wait for the command
        
    Returns:
        subprocess.CompletedProcess object
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    # Disable wandb for testing - use offline mode
    env["WANDB_MODE"] = "disabled"
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
        env=env
    )
    return result


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_training_")
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestDatasetAvailability:
    """Test that required datasets exist."""
    
    def test_mimic_dataset_exists(self):
        """Test MIMIC dataset directory exists."""
        assert os.path.exists(DATASETS["MIMIC"]), \
            f"MIMIC dataset not found at {DATASETS['MIMIC']}"
        assert os.path.exists(os.path.join(DATASETS["MIMIC"], "train", "listfile.csv")), \
            "MIMIC train listfile not found"
        assert os.path.exists(os.path.join(DATASETS["MIMIC"], "test", "listfile.csv")), \
            "MIMIC test listfile not found"
    
    def test_inalo_dataset_exists(self):
        """Test INALO dataset directory exists."""
        assert os.path.exists(DATASETS["INALO"]), \
            f"INALO dataset not found at {DATASETS['INALO']}"
        assert os.path.exists(os.path.join(DATASETS["INALO"], "train", "listfile.csv")), \
            "INALO train listfile not found"
        assert os.path.exists(os.path.join(DATASETS["INALO"], "test", "listfile.csv")), \
            "INALO test listfile not found"


class TestDeterministicTraining:
    """Test deterministic training for all models and datasets."""
    
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("dataset_name,dataset_path", [
        ("MIMIC", DATASETS["MIMIC"]),
        ("INALO", DATASETS["INALO"]),
    ])
    def test_deterministic_training(self, model, dataset_name, dataset_path, temp_output_dir):
        """
        Test deterministic training with specified model and dataset.
        Uses --small_part to limit data and --num_trials 1 for speed.
        """
        if not os.path.exists(dataset_path):
            pytest.skip(f"Dataset {dataset_name} not available at {dataset_path}")
        
        cmd = [
            VENV_PYTHON,
            DETERMINISTIC_SCRIPT,
            "--network", f"ptsa/models/deterministic/{model}.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", model,
            "--model_name", f"{model}_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=300)
        
        # Print output for debugging if test fails
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, \
            f"Deterministic {model} training on {dataset_name} failed:\n{result.stderr}"
        
        # Check that output files were created
        expected_dataset_dir = os.path.basename(dataset_path)
        output_model_dir = os.path.join(
            temp_output_dir, expected_dataset_dir, "deterministic", model
        )
        
        # Check for results table
        assert os.path.exists(os.path.join(output_model_dir, "results_table.csv")), \
            f"Results CSV not created for {model} on {dataset_name}"
        assert os.path.exists(os.path.join(output_model_dir, "results_table.txt")), \
            f"Results TXT not created for {model} on {dataset_name}"
        assert os.path.exists(os.path.join(output_model_dir, "best_hyperparams.txt")), \
            f"Best hyperparams not created for {model} on {dataset_name}"


class TestProbabilisticTraining:
    """Test probabilistic training for all models and datasets."""
    
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("dataset_name,dataset_path", [
        ("MIMIC", DATASETS["MIMIC"]),
        ("INALO", DATASETS["INALO"]),
    ])
    def test_probabilistic_training(self, model, dataset_name, dataset_path, temp_output_dir):
        """
        Test probabilistic training with specified model and dataset.
        Uses --small_part to limit data and --num_trials 1 for speed.
        """
        if not os.path.exists(dataset_path):
            pytest.skip(f"Dataset {dataset_name} not available at {dataset_path}")
        
        cmd = [
            VENV_PYTHON,
            PROBABILISTIC_SCRIPT,
            "--network", f"ptsa/models/probabilistic/{model}_classification.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", model,
            "--model_name", f"{model}_probabilistic_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=600)
        
        # Print output for debugging if test fails
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, \
            f"Probabilistic {model} training on {dataset_name} failed:\n{result.stderr}"
        
        # Check that output files were created
        expected_dataset_dir = os.path.basename(dataset_path)
        output_model_dir = os.path.join(
            temp_output_dir, expected_dataset_dir, "probabilistic", model
        )
        
        # Check for results table
        assert os.path.exists(os.path.join(output_model_dir, "results_table.csv")), \
            f"Results CSV not created for {model} on {dataset_name}"
        assert os.path.exists(os.path.join(output_model_dir, "results_table.txt")), \
            f"Results TXT not created for {model} on {dataset_name}"
        assert os.path.exists(os.path.join(output_model_dir, "best_hyperparams.txt")), \
            f"Best hyperparams not created for {model} on {dataset_name}"


class TestModelImports:
    """Test that all model modules can be imported correctly."""
    
    @pytest.mark.parametrize("model", MODELS)
    def test_deterministic_model_import(self, model):
        """Test deterministic model can be imported."""
        if model == "transformer":
            from ptsa.models.deterministic.transformer_classification import TransformerIHM
            assert TransformerIHM is not None
        else:
            module = __import__(
                f"ptsa.models.deterministic.{model}_classification",
                fromlist=[model.upper()]
            )
            model_class = getattr(module, model.upper())
            assert model_class is not None
    
    @pytest.mark.parametrize("model", MODELS)
    def test_probabilistic_model_import(self, model):
        """Test probabilistic model can be imported."""
        if model == "transformer":
            from ptsa.models.probabilistic.transformer_classification import TransformerIHM
            assert TransformerIHM is not None
        else:
            module = __import__(
                f"ptsa.models.probabilistic.{model}_classification",
                fromlist=[model.upper()]
            )
            model_class = getattr(module, model.upper())
            assert model_class is not None


class TestProbabilisticModelUncertainty:
    """Test that probabilistic models return correct uncertainty values."""
    
    @pytest.mark.parametrize("model", MODELS)
    def test_predict_with_uncertainty_returns_three_values(self, model):
        """Test that predict_with_uncertainty returns mean, epistemic, and aleatoric."""
        import torch
        
        if model == "transformer":
            from ptsa.models.probabilistic.transformer_classification import TransformerIHM
            model_instance = TransformerIHM(
                input_size=38, d_model=64, nhead=2, num_layers=1, dropout=0.2
            )
        elif model == "lstm":
            from ptsa.models.probabilistic.lstm_classification import LSTM
            model_instance = LSTM(input_size=38, hidden_size=64, num_layers=1, dropout=0.2)
        elif model == "rnn":
            from ptsa.models.probabilistic.rnn_classification import RNN
            model_instance = RNN(input_size=38, hidden_size=64, num_layers=1, dropout=0.2)
        elif model == "gru":
            from ptsa.models.probabilistic.gru_classification import GRU
            model_instance = GRU(input_size=38, hidden_size=64, num_layers=1, dropout=0.2)
        
        # Create dummy input: (batch_size=1, seq_len=48, features=38)
        dummy_input = torch.randn(1, 48, 38)
        
        # Get prediction with uncertainty
        result = model_instance.predict_with_uncertainty(dummy_input, num_samples=10)
        
        # Should return 3 values: mean, epistemic_variance, aleatoric_variance
        assert len(result) == 3, \
            f"Expected 3 return values (mean, epistemic, aleatoric), got {len(result)}"
        
        mean, epistemic_var, aleatoric_var = result
        
        # Check shapes
        assert mean.shape == torch.Size([1]), f"Mean shape mismatch: {mean.shape}"
        assert epistemic_var.shape == torch.Size([1]), \
            f"Epistemic variance shape mismatch: {epistemic_var.shape}"
        assert aleatoric_var.shape == torch.Size([1]), \
            f"Aleatoric variance shape mismatch: {aleatoric_var.shape}"
        
        # Check values are valid (non-negative for variances, 0-1 for mean probability)
        assert (mean >= 0).all() and (mean <= 1).all(), \
            f"Mean probability out of range [0,1]: {mean}"
        assert (epistemic_var >= 0).all(), \
            f"Epistemic variance should be non-negative: {epistemic_var}"
        assert (aleatoric_var >= 0).all(), \
            f"Aleatoric variance should be non-negative: {aleatoric_var}"


class TestOutputDirectoryStructure:
    """Test that output directory structure is created correctly."""
    
    @pytest.mark.slow
    def test_deterministic_output_structure(self, temp_output_dir):
        """Test deterministic training creates correct directory structure."""
        dataset_path = DATASETS["MIMIC"]
        if not os.path.exists(dataset_path):
            pytest.skip("MIMIC dataset not available")
        
        cmd = [
            VENV_PYTHON,
            DETERMINISTIC_SCRIPT,
            "--network", "ptsa/models/deterministic/lstm.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", "lstm",
            "--model_name", "lstm_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=300)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        
        # Check directory structure
        expected_dir = os.path.join(
            temp_output_dir, "in-hospital-mortality-fixed", "deterministic", "lstm"
        )
        assert os.path.exists(expected_dir), \
            f"Expected directory structure not created: {expected_dir}"
    
    @pytest.mark.slow
    def test_probabilistic_output_structure(self, temp_output_dir):
        """Test probabilistic training creates correct directory structure."""
        dataset_path = DATASETS["MIMIC"]
        if not os.path.exists(dataset_path):
            pytest.skip("MIMIC dataset not available")
        
        cmd = [
            VENV_PYTHON,
            PROBABILISTIC_SCRIPT,
            "--network", "ptsa/models/probabilistic/lstm_classification.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", "lstm",
            "--model_name", "lstm_probabilistic_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=600)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        
        # Check directory structure
        expected_dir = os.path.join(
            temp_output_dir, "in-hospital-mortality-fixed", "probabilistic", "lstm"
        )
        assert os.path.exists(expected_dir), \
            f"Expected directory structure not created: {expected_dir}"


class TestResultsTableContent:
    """Test that results tables have correct content."""
    
    @pytest.mark.slow
    def test_deterministic_results_table_columns(self, temp_output_dir):
        """Test deterministic results table has correct columns including uncertainty placeholders."""
        import pandas as pd
        
        dataset_path = DATASETS["MIMIC"]
        if not os.path.exists(dataset_path):
            pytest.skip("MIMIC dataset not available")
        
        cmd = [
            VENV_PYTHON,
            DETERMINISTIC_SCRIPT,
            "--network", "ptsa/models/deterministic/lstm.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", "lstm",
            "--model_name", "lstm_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=300)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        
        csv_path = os.path.join(
            temp_output_dir, "in-hospital-mortality-fixed", "deterministic", "lstm",
            "results_table.csv"
        )
        
        df = pd.read_csv(csv_path)
        
        expected_columns = [
            'Model', 'Dataset', 'AUROC', 'AUPRC', 'Precision', 'Recall', 
            'F1-Score', 'Aleatoric Uncertainty', 'Epistemic Uncertainty'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check deterministic model has "–" for uncertainties
        assert df['Aleatoric Uncertainty'].iloc[0] == '–', \
            "Deterministic model should have '–' for Aleatoric Uncertainty"
        assert df['Epistemic Uncertainty'].iloc[0] == '–', \
            "Deterministic model should have '–' for Epistemic Uncertainty"
        
        # Check dataset name
        assert df['Dataset'].iloc[0] == 'MIMIC', \
            f"Expected dataset 'MIMIC', got {df['Dataset'].iloc[0]}"
    
    @pytest.mark.slow
    def test_probabilistic_results_table_columns(self, temp_output_dir):
        """Test probabilistic results table has correct columns with uncertainty values."""
        import pandas as pd
        
        dataset_path = DATASETS["MIMIC"]
        if not os.path.exists(dataset_path):
            pytest.skip("MIMIC dataset not available")
        
        cmd = [
            VENV_PYTHON,
            PROBABILISTIC_SCRIPT,
            "--network", "ptsa/models/probabilistic/lstm_classification.py",
            "--partition", "custom",
            "--data", dataset_path,
            "--model", "lstm",
            "--model_name", "lstm_probabilistic_test.pth",
            "--output_dir", temp_output_dir,
            "--num_trials", "1",
            "--small_part",
        ]
        
        result = run_training_command(cmd, timeout=600)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        
        csv_path = os.path.join(
            temp_output_dir, "in-hospital-mortality-fixed", "probabilistic", "lstm",
            "results_table.csv"
        )
        
        df = pd.read_csv(csv_path)
        
        expected_columns = [
            'Model', 'Dataset', 'AUROC', 'AUPRC', 'Precision', 'Recall', 
            'F1-Score', 'Aleatoric Uncertainty', 'Epistemic Uncertainty'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check probabilistic model has numeric values for uncertainties
        aleatoric = df['Aleatoric Uncertainty'].iloc[0]
        epistemic = df['Epistemic Uncertainty'].iloc[0]
        
        # Should be numeric strings, not "–"
        assert aleatoric != '–', \
            "Probabilistic model should have numeric Aleatoric Uncertainty"
        assert epistemic != '–', \
            "Probabilistic model should have numeric Epistemic Uncertainty"
        
        # Should be parseable as floats
        try:
            float(aleatoric)
            float(epistemic)
        except ValueError:
            pytest.fail(f"Uncertainty values not numeric: {aleatoric}, {epistemic}")
        
        # Check model name includes "+ MC Dropout"
        assert "+ MC Dropout" in df['Model'].iloc[0], \
            f"Expected model name with '+ MC Dropout', got {df['Model'].iloc[0]}"
        
        # Check dataset name
        assert df['Dataset'].iloc[0] == 'MIMIC', \
            f"Expected dataset 'MIMIC', got {df['Dataset'].iloc[0]}"


# Quick smoke tests - run these first for fast feedback
class TestSmoke:
    """Quick smoke tests to verify basic functionality."""
    
    def test_scripts_exist(self):
        """Test that training scripts exist."""
        assert os.path.exists(DETERMINISTIC_SCRIPT), \
            f"Deterministic script not found: {DETERMINISTIC_SCRIPT}"
        assert os.path.exists(PROBABILISTIC_SCRIPT), \
            f"Probabilistic script not found: {PROBABILISTIC_SCRIPT}"
    
    def test_venv_python_exists(self):
        """Test that virtual environment Python exists or system Python is available."""
        venv_path = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
        if not os.path.exists(venv_path):
            # Fallback to system python
            assert sys.executable is not None, "No Python interpreter available"
        else:
            assert os.path.exists(venv_path), f"Virtual env Python not found: {venv_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
