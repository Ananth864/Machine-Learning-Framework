import json
import logging
import importlib
import os
import shutil
from pathlib import Path
import pandas as pd

ROOT_DIR = Path("main.py").resolve().parent.parent
OPERATORS = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x / y}

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def json_parse(config_path):
    """Parse and load JSON config file."""
    with open(config_path) as f:
        return json.load(f)

def add_columns(new_columns, location):
    """Add new columns to training features JSON file if not already present."""
    with open(location, 'r') as f:
        data = json.load(f)
        features = data['training_features']
        
    features.extend([item for item in new_columns if item not in features])
    
    with open(location, "w") as f:
        json.dump({'training_features': features}, f)

def remove_invalid_columns(column_sets, dataframe):
    """Remove columns not present in dataframe from column sets."""
    valid_columns = []
    removed_columns = []

    for column_set in column_sets:
        valid_set = [col for col in column_set if col in dataframe.columns]
        removed_set = [col for col in column_set if col not in dataframe.columns]
        
        if valid_set:
            valid_columns.append(valid_set)
        removed_columns.extend(removed_set)

    if removed_columns:
        input(f"Removed invalid columns: {removed_columns}. Press enter to continue:")
    
    return valid_columns

def operator(op1, op, op2):
    """Apply arithmetic operator to operands."""
    if op not in OPERATORS:
        raise ValueError(f"Unsupported operator: {op}")
    return OPERATORS[op](op1, op2)

def make_folders(project, experiment, experiment_type):
    """Create folder structure for new experiment run."""
    root_dir = Path(ROOT_DIR)
    project_path = root_dir / 'models' / project / experiment / experiment_type
    
    os.makedirs(project_path, exist_ok=True)
    
    existing_runs = [f for f in os.listdir(project_path) if f.startswith("run_")]
    run_num = max([int(r.split('_')[1]) for r in existing_runs], default=-1) + 1
    run_name = f"run_{run_num}"
    run_path = project_path / run_name
    os.mkdir(run_path)
    
    shutil.copy(root_dir / 'resources/Project_Configs.json', run_path)
    for subdir in ['model', 'features', 'encodings']:
        os.makedirs(run_path / subdir)
    
    output_path = root_dir / 'output_dir/train_results' / project / experiment / experiment_type / run_name
    if output_path.exists():
        shutil.rmtree(output_path)
    
    for subdir in ['datasets', 'shap_plots/bar_plot', 'shap_plots/beeswarm', 'shap_plots/heatmap']:
        os.makedirs(output_path / subdir)
    
    return run_name

def check_model_scoring_input(scoring_config):
    """Validate model paths in scoring config."""
    for key, model_path in scoring_config["Scoring"].items():
        if not model_path.endswith('.pkl'):
            model_path += '.pkl'
        if not os.path.exists(model_path):
            logging.error(f"Incorrect model location provided for {key}")
            raise ValidationError

def get_model_list():
    """Get list of models and their feature sets from scoring config."""
    scoring_path = ROOT_DIR / 'resources/model_scoring.json'
    with open(scoring_path, 'r+') as f:
        config = json.load(f)
        model_list = []
        set_list = []

        for key, model_path in config["Scoring"].items():
            parts = model_path.split('_')
            set_num = int(parts[-2])
            set_list.append(set_num)

            if model_path.endswith('.pkl'):
                model_path = model_path[:-4]
                config["Scoring"][key] = model_path
            model_list.append(model_path)

        check_model_scoring_input(config)
        
        f.seek(0)
        json.dump(config, f)
        f.truncate()

    return model_list, set_list

def get_run(model_path):
    """Extract run info from model path."""
    path = Path(model_path)
    run_dir = path.parent.parent
    model_type = path.parent.parent.parent.name
    year = path.parent.parent.parent.parent.name
    return run_dir, model_type, year

def importvar(var, fname, notfound):
    """Import variable from module with fallback value."""
    try:
        return getattr(importlib.import_module(fname), var)
    except:
        return notfound

def making_rankings_train(df):
    """Create ranked training results dataframe sorted by validation MAE."""
    val_results = df[df['Split'] == 'CV-Val'].sort_values(by='MAE')
    sorted_indices = val_results.index

    df_sorted = pd.DataFrame(columns=df.columns)
    for idx in sorted_indices:
        pair_rows = df.loc[idx-1:idx] 
        df_sorted = pd.concat([df_sorted, pair_rows])
        df_sorted = pd.concat([df_sorted, pd.DataFrame(columns=df.columns, index=[len(df_sorted)])])

    df_sorted.reset_index(drop=True, inplace=True)
    df_sorted.insert(0, 'Rank', [f'Rank_{i//2+1}' for i in range(len(df_sorted))])
    
    return df_sorted
    