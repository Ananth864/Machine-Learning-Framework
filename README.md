# ML Framework

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![PySpark](https://img.shields.io/badge/pyspark-3.0%2B-orange)]()
[![PyCaret](https://img.shields.io/badge/pycaret-latest-green)]()

## Overview

A standardized pipeline for efficient development, deployment, and monitoring of machine learning projects. The framework incorporates most features required for model development while maintaining generalizability.

### Projects & Experiments Structure

The framework uses a top-level organization where each project is a complete configuration unit:

- **Project**: A top-level ML initiative with its own complete configuration
  - Contains target variable definition
  - Specifies experiment type (regression/classification)
  - Defines data sources
  - Sets feature engineering and selection parameters
  - Example projects: `blockview`, `customer_churn`, `fraud_detection`

- **Experiment**: Variations of model approaches within a project
  - Uses project's feature engineering and data configurations
  - Can specify different models and optimization metrics
  - Example: Project `blockview` might have experiments:
    - `exp1`: Using linear models for baseline
    - `exp2`: Testing ensemble methods
    - `exp3`: Exploring deep learning approaches

Benefits of this structure:
- **Reusability**: Feature engineering and configurations can be shared across experiments
- **Organization**: Clear separation of concerns between different ML initiatives
- **Maintainability**: Easy to track and update related experiments
- **Scalability**: New experiments can be added without affecting existing ones
- **Versioning**: Each experiment maintains its own run history
- **Flexibility**: Different modeling approaches can coexist within the same project

### Key Benefits
- Reduces development time by eliminating code rewriting
- Standardized pipeline with high maintainability
- Improves model versioning and tracking
- Eases project understanding and implementation
- Enables easy model improvement and iteration

## Repository Structure

```
ML_Framework/
├── main.py                     # Main entry point for the framework
├── Feature_Engineering.py      # Feature engineering functionality
├── Data_prep.py                # Data preparation and imputation
├── Train_Class.py              # Model training functionality
├── Test_Class.py               # Model testing and evaluation
├── Score.py                    # Model scoring for production
├── Project_Config_Parser.py    # Configuration parser
├── utils.py                    # Utility functions
├── Hive_Class.py               # Data import from Hive (optional)
├── Custom_User_Functions.py    # Custom user-defined functions
├── resources/                  # Resource files
│   ├── Project_Configs.json    # Project configuration
│   ├── Training_Columns.json   # Training feature sets
│   └── Model_Scoring.json      # Scoring configuration
├── models/                     # Saved models
│   └── [project]/[experiment]/[experiment_type]/[run_x]/
│       ├── model/              # Saved model files
│       ├── features/           # Feature configurations
│       └── encodings/          # Encoding mappings
└── output_dir/                 # Output directory
    ├── logging/                # Log files
    └── train_results/          # Training and test results
        └── [project]/[experiment]/[experiment_type]/[run_x]/
            ├── datasets/       # Generated datasets
            ├── shap_plots/     # SHAP explanations
            └── [metrics files] # Performance metrics
```

## Pipeline Flow
```
Config → Remote/Local Data Extraction → Feature Engineering → Custom Feature Engineering → 
Imputation → OneHotEncoding → Feature Selection → Training → Testing
```

## Usage

### Command Line
```bash
python main.py project experiment
```

Arguments:
- `project`: Project name containing experiment types
- `experiment`: Individual experiment type within the project
Example: `python main.py blockview exp1`

## Configuration Guide

### Project Config Structure
```json
{
    "project_name": {
        "target": "target_column_name",
        "experiment_type": "regression|classification",
        "data": {
            "data_retrieval_mode": "remote|local",
            "local": {
                "train_table_path": "resources/train.csv",
                "test_table_path": "resources/test.csv"
            },
            "remote": {
                "train_db": "database_name",
                "test_db": "database_name",
                "train_table": "table_name",
                "test_table": "table_name"
            }
        },
        "feature_engineering": {},
        "feature_selection": {},
        "parameters": {},
        "fold": 5,
        "search_algorithm": "string",
        "search_library": "string",
        "setup_params": {},
        "imputation_method": "string",
        "custom_metrics": [],
        "experiments": {},
        "Top_N_Models": 3,
        "Ensemble_Models": {},
        "Stack_Models": {},
        "Blend_Models": false,
        "ensemble_scoring": {
            "flag": 0,
            "method": {
                "name": "",
                "fn": ""
            }
        }
    }
}
```

### Feature Engineering Configuration

#### Numerical Feature Engineering
```json
"feature_engineering": {
    "numerical_feature": {
        "calculation": "column_1/column_2",  // Supports +,-,*,/
        "fn": "log|sqrt|none"
    }
}
```

The `calculation` parameter specifies operation to perform on columns, supporting basic arithmetic operations (+, -, *, /). 
The `fn` parameter applies a non-linear function to the result:
- `log`: Natural logarithm
- `sqrt`: Square root
- `none`: No transformation

#### Categorical Feature Engineering
```json
"feature_engineering": {
    "categorical_feature": {
        "encoding": "minority-group|one-hot",
        "threshold": 2,  // Minimum frequency threshold
        "max_dummy_feat_cnt": 10  // Maximum categories to keep
    }
}
```

For categorical features:
- `minority-group`: Consolidates low-frequency categories into "other"
- `one-hot`: Creates binary indicator columns for each category

### Feature Selection Configuration
```json
"feature_selection": {
    "include": true,
    "method": "top|coverage|cut-off|boruta",
    "model": "dt|rf|xgb",
    "mul_col_thr": 0.4,  // Maximum collinearity threshold
    "variance_threshold": 0.0,  // Minimum variance threshold
    "cut-off": 0.01,  // Feature importance cutoff
    "top_n": 10,  // Number of top features to select
    "threshold": 0.99  // Cumulative importance threshold
}
```

Method options:
- `top`: Select top N features by importance
- `coverage`: Select features whose cumulative importance reaches threshold
- `cut-off`: Select features above importance threshold
- `boruta`: Use Boruta algorithm for feature selection

### Model Parameters Configuration
```json
"parameters": {
    "model_name": {
        "param1": [value1, value2, value3],  // Parameter search space
        "param2": [value1, value2, value3]
    }
}
```

Example for Huber regressor:
```json
"huber": {
    "epsilon": [1.25, 1.3, 1.35, 1.4, 1.45],
    "max_iter": [80, 90, 100, 110, 120],
    "alpha": [0.00005, 0.000075, 0.0001, 0.000125, 0.00015]
}
```

Note: Parameters must be in list format. Invalid parameters will abort training and use initial weights.

### Search Configuration
```json
"search_algorithm": "random|grid|bayesian|hyperopt|optuna|bohb",
"search_library": "scikit-learn|scikit-optimize|tune-sklearn|optuna"
```

Available combinations:
- scikit-learn: random, grid
- scikit-optimize: bayesian
- tune-sklearn: random, grid, bayesian, hyperopt, optuna, bohb
- optuna: random, tpe

### Imputation Configuration
```json
"imputation_method": "mean|median|most_frequent|knn"
```

The framework handles imputation differently based on column type:
- Numeric columns: Uses specified method (mean, median, etc.)
- Categorical columns: Always uses most frequent value (mode)

### Custom Metrics Configuration
```json
"custom_metrics": ["explained_variance_score", "custom_metric_name"]
```

Define custom metrics in `Custom_User_Functions.py`:
```python
def custom_metric(y_actual, y_pred):
    # Metric calculation
    return metric_value  # Must return pandas.Series
```

Default metrics provided: MAE, MSE, RMSE, R2, RMSLE, MAPE

### Model Ensemble Configuration
```json
"Top_N_Models": 3,  // Top N models for ensemble
"Ensemble_Models": {
    "ensemble_flag": true,
    "method": "Bagging|Boosting",
    "n_estimators": 10
},
"Stack_Models": {
    "stack_flag": true,
    "meta_model": "model_name",
    "restack": false
},
"Blend_Models": true
```

- `Top_N_Models`: Number of best-performing models to include in ensembles
- `Ensemble_Models`: Configure bagging or boosting ensembles
- `Stack_Models`: Configure stacked ensemble with meta-learner
- `Blend_Models`: Enable/disable model blending

### Experiments Configuration
```json
"experiments": {
    "experiment_name": {
        "models": ["lr", "xgboost", "lightgbm"],  // or single string
        "n_iter": 100,
        "optimize": "MAE"
    }
}
```

Each experiment can specify:
- `models`: List of models to train (or single model as string)
- `n_iter`: Number of hyperparameter tuning iterations
- `optimize`: Metric to optimize during training

## Available Models

### Regression Models
- Linear Models: `lr`, `lasso`, `ridge`, `en`, `lar`, `llar`, `omp`, `br`, `ard`, `par`, `ransac`, `tr`, `huber`, `kr`
- Tree-based: `dt`, `rf`, `et`, `ada`, `gbr`
- Advanced: `svm`, `knn`, `mlp`, `xgboost`, `lightgbm`, `catboost`

### Classification Models
- Linear Models: `lr`, `ridge`
- Tree-based: `dt`, `rf`, `et`, `ada`, `gbc`
- Advanced: `svm`, `rbfsvm`, `gpc`, `mlp`, `qda`, `lda`, `xgboost`, `lightgbm`, `catboost`, `nb`, `knn`

## Custom Feature Engineering

Modify `Custom_User_Functions.py`:
```python
def custom_feat(project, experiment, df_train, df_test):
    # Custom feature engineering
    return df_train, df_test

def import_remote_data(train_db, train_table, test_db, test_table):
    # Custom data import
    return df_train, df_test
```

The `custom_feat` function is called after standard feature engineering and allows for project-specific transformations.

## Training Column Sets
Define in `Training_Columns.json`:
```json
{
    "training_features": [
        ["feature1", "feature2"],
        ["feature1", "feature3", "feature4"]
    ]
}
```

If empty (`"training_features": [[]]`), the framework uses all available features.

## Output Directory Structure

```
output_dir/
├── logging/
│   └── log_YYYY_MM_DD_HH_MM_SS.log
└── train_results/
    └── project_name/
        └── experiment_name/
            └── experiment_type/
                └── run_n/
                    ├── datasets/
                    │   ├── train_set_1_dataset.csv
                    │   └── test_set_1_dataset.csv
                    ├── measure_name_set_m_model_n.csv
                    ├── measure_name_set_mapping.json
                    ├── measure_name_train_combined.csv
                    ├── measure_name_test_combined.csv
                    └── shap_plots/
                        ├── bar_plot/
                        ├── beeswarm/
                        └── heatmap/
```

The `train_results` directory contains:
- Dataset files for each feature set
- Performance metrics for each model-feature set combination
- Feature set mappings
- Combined performance metrics tables
- SHAP plots for model interpretability

## Models Directory Structure

```
models/
└── project_name/
    └── experiment_name/
        └── experiment_type/
            └── run_n/
                ├── model/
                │   └── experiment_type_project_model_set_i_model_name
                ├── features/
                │   ├── experiment_type_project_features.json
                │   └── experiment_type_project_set_mapping.json
                └── encodings/
                    ├── categorical_columns.json
                    └── feature_name_minority_encoded.json
```

The `models` directory stores:
- Trained model files (PyCaret format)
- Feature definitions and mappings
- Encoding information for categorical variables

## Logging

Logs are stored in: `output_dir/logging/log_YYYY_MM_DD_HH_MM_SS.log`

## SHAP Visualization

The framework automatically generates model explanations using SHAP:
- Bar plots: Overall feature importance
- Beeswarm plots: Distribution of SHAP values
- Heatmap plots: Feature interactions

## Installation

Required packages:
```
pandas
numpy
scikit-learn
pyspark
pycaret
xgboost
lightgbm
catboost
shap
matplotlib
findspark
```

For advanced search algorithms:
```bash
pip install scikit-optimize
pip install tune-sklearn ray[tune]
pip install hyperopt
pip install optuna
pip install hpbandster ConfigSpace
```

