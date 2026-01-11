# Thesis Experiment: Log Generation and Experiment Process (`exp1.py`)

This document describes the process of log generation and experimentation as implemented in the `exp1.py` file of this project.

## Overview
The `exp1.py` script is designed to automate the process of running experiments and generating logs for analysis. It is a core part of the thesis experiments, enabling reproducibility and systematic evaluation of different configurations or models.


## Step-by-Step Process of the First Experiment

Below is a thorough, step-by-step explanation of the process followed in the first experiment (`exp1.py`):

1. **Initialization**
    - The script begins by importing all necessary libraries (e.g., numpy, pandas, torch, sklearn, etc.).
    - It sets random seeds for reproducibility, ensuring that results can be replicated.
    - Configuration parameters are loaded, either from a configuration file or defined directly in the script. These include dataset paths, model types, hyperparameters, and output directories.
    - Output directories for logs and results are created if they do not already exist, organizing experiment outputs.

2. **Experiment Configuration**
    - The script defines or loads the specific parameters for the experiment, such as:
        - Which dataset to use (e.g., path, format, preprocessing steps)
        - Model architecture and type (e.g., neural network, decision tree)
        - Hyperparameters (e.g., learning rate, batch size, number of epochs)
        - Number of experiment runs or folds (for cross-validation)
    - These parameters are either hardcoded, loaded from a config file, or parsed from command-line arguments.

3. **Data Preparation**
    - The dataset is loaded from the specified source.
    - Preprocessing steps are applied, such as normalization, encoding categorical variables, splitting into train/test sets, or shuffling.
    - If required, data is checked for integrity and completeness.

4. **Degradation Process (if applicable)**
    - If the experiment involves testing robustness, the script applies degradation or corruption to the data or model. This can include:
        - Adding noise to input features or labels
        - Removing or masking certain features
        - Simulating faults or missing data
    - The type and severity of degradation are controlled by configuration parameters, allowing systematic evaluation of different conditions.

5. **Execution Loop**
    - The script enters a loop to perform the experiment multiple times (for different configurations, random seeds, or cross-validation folds).
    - For each run:
        - The model or algorithm is initialized with the current configuration.
        - Training is performed on the prepared (and possibly degraded) dataset.
        - Evaluation is conducted on a validation or test set, collecting metrics such as accuracy, loss, precision, recall, etc.
        - Any errors or exceptions are caught and logged for debugging and reproducibility.

6. **Result Collection and Logging**
    - After each run, the script collects all relevant outputs, including:
        - Final model performance metrics
        - Configuration details for the run (parameters, random seed, degradation type/level)
        - Timestamps and unique run identifiers
        - Any intermediate results or checkpoints, if configured
        - Errors or exceptions encountered during the run
    - All this information is logged in a structured format (e.g., CSV, JSON, or plain text) and saved in the designated output directory.

7. **Log Storage and Organization**
    - The script ensures that logs are organized by experiment, run, or configuration, making it easy to retrieve and analyze results later.
    - Log files are named systematically (e.g., including timestamps or configuration hashes) to avoid overwriting and to facilitate tracking.

8. **(Optional) Analysis and Visualization**
    - Optionally, the script may include code to aggregate results across runs, compute summary statistics, or generate plots/visualizations for further analysis.
    - These analyses help in interpreting the experiment outcomes and drawing conclusions.

## Experiment and Degradation Process
- **Setup**: Import necessary libraries and define experiment parameters.
- **Data Preparation**: Load and preprocess datasets as required.
- **Degradation (if used)**: Apply degradation or corruption to the data or model to simulate challenging conditions and test robustness.
- **Model/Algorithm Execution**: Run the core experiment logic (e.g., training, evaluation).
- **Result Collection**: Gather all relevant outputs and metrics.
- **Logging**: Write results and metadata to log files for each experiment run.
- **Analysis (Optional)**: The script may include code to aggregate or visualize results from the logs.

## How to Run
1. Ensure all dependencies are installed (see `requirements.txt` if available).
    ```bash
    pip install requirements.txt
    ```
2. Run the script using Python:
   ```bash
   python exp1.py
   ```
3. Logs and results will be generated in the specified output directory.

## Notes
- Modify the configuration section in `exp1.py` to adjust experiment parameters.
- Check the output/log directory for generated logs after execution.
- For detailed analysis, use the logs as input to further scripts or tools.

---
For further details, refer to the comments and documentation within `exp1.py` itself.
