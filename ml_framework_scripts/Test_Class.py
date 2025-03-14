import logging
from pathlib import Path
import pandas as pd
import json
import os
import shap
import matplotlib.pyplot as plt
from pycaret.regression import *

ROOT_DIR = Path("main.py").resolve().parent.parent

class Test:
    """Class for testing ML models and generating SHAP explanations."""

    def __init__(self, project, experiment, experiment_type, project_config, target):
        """Initialize Test class.

        Args:
            project: Project name
            experiment: Experiment name
            experiment_type: Type of ML experiment (regression/classification) 
            project_config: Configuration dictionary
            target: Target variable name
        """
        self.experiment = experiment
        self.experiment_type = experiment_type
        self.project = project
        self.project_config = project_config
        self.target = target
        self.train_config = project_config['experiments'][str(experiment)]

    def _generate_shap_plots(self, model, data, training_features, j, run, algo):
        """Generate SHAP plots for model explanation.
        
        Args:
            model: Trained model object
            data: Test data
            training_features: List of feature names
            j: Feature set index
            run: Run identifier
            algo: Algorithm name
        """
        try:
            explainer = self._create_explainer(model, data[training_features])
            if not explainer:
                return
            
            shap_values = explainer(data[training_features])
            self._save_shap_plots(shap_values, j, run, algo)
        except Exception as e:
            logging.warning(f"Failed to generate SHAP plots for {algo}: {str(e)}")

    def _create_explainer(self, model, data):
        """Create SHAP explainer based on model type.
        
        Args:
            model: Trained model
            data: Test data
            
        Returns:
            SHAP explainer object
        """
        try:
            return shap.LinearExplainer(model.named_steps["trained_model"], data)
        except:
            try:
                return shap.TreeExplainer(model[len(model.steps)-1], data, feature_perbutation="correlation dependent")
            except:
                return None

    def _save_shap_plots(self, shap_values, j, run, algo):
        """Save generated SHAP plots to files.
        
        Args:
            shap_values: SHAP values for test data
            j: Feature set index
            run: Run identifier
            algo: Algorithm name
        """
        plots = {
            'bar': shap.plots.bar,
            'beeswarm': shap.plots.beeswarm,
            'heatmap': shap.plots.heatmap
        }

        for plot_type, plot_func in plots.items():
            try:
                plot_func(shap_values)
                plt.savefig(
                    ROOT_DIR / f'output_dir/train_results/{self.project}/{self.experiment}/{self.experiment_type}/{run}/shap_plots/{plot_type}/{self.experiment_type}_{self.project}_{plot_type}_set_{j}_{algo}.png',
                    bbox_inches='tight'
                )
                plt.close()
            except Exception as e:
                logging.warning(f"{plot_type} plot failed for {algo}: {str(e)}")

    def _test_single_model(self, model_path, training_features, data_to_test, run):
        """Test a single trained model.
        
        Args:
            model_path: Path to saved model file
            training_features: List of feature names 
            data_to_test: Test data DataFrame
            run: Run identifier
            
        Returns:
            Dict containing test results and feature set info
        """
        model_info = self._parse_model_filename(model_path)
        if not model_info:
            return None

        set_idx, algo = model_info
        feature_set = training_features[set_idx-1]
        test_columns = feature_set + [self.target]

        try:
            test_data = data_to_test[test_columns]
            # Remove .pkl if it exists before loading
            model_path_str = str(model_path)
            if model_path_str.endswith('.pkl'):
                model_path_str = model_path_str[:-4]
            model = load_model(model_path_str)
            predictions = predict_model(model, data=test_data)
            
            test_result = pull()
            test_result["Model"] = algo
            
            self._generate_shap_plots(model, test_data, feature_set, set_idx, run, algo)
            
            return {'result': test_result, 'set': f'set_{set_idx}'}
        except Exception as e:
            logging.error(f"Error testing model {model_path}: {str(e)}")
            return None

    def _parse_model_filename(self, model_path):
        """Parse model filename to get feature set index and algorithm.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Tuple of (set index, algorithm name)
        """
        path_str = str(model_path)
        if not path_str.endswith('.pkl'):
            path_str = path_str[:-4]
            
        splits = path_str.split('_')
        try:
            set_idx = int(splits[-2])
            algo = splits[-1]
            return set_idx, algo
        except:
            logging.error(f"Failed to parse model filename: {model_path}")
            return None

    def test_feat_set(self, data_to_test, run):
        """Test all models from a training run.
        
        Args:
            data_to_test: Test data DataFrame
            run: Run identifier
        """
        features_path = ROOT_DIR / f'models/{self.project}/{self.experiment}/{self.experiment_type}/{run}/features/{self.experiment_type}_{self.project}_features.json'
        
        with features_path.open('r') as f:
            training_features = json.load(f)['training_features']

        model_dir = ROOT_DIR / f'models/{self.project}/{self.experiment}/{self.experiment_type}/{run}/model'
        results = []
        
        for model_path in model_dir.glob('*'):
            if model_path.is_file():
                result = self._test_single_model(model_path, training_features, data_to_test, run)
                if result:
                    results.append(result)

        if not results:
            combined_results = pd.DataFrame()
        else:
            combined_results = pd.concat([r['result'] for r in results], ignore_index=True)
            combined_results.insert(0, "Set", [r['set'] for r in results])
            combined_results.sort_values(by=["MAE"], inplace=True)
            combined_results.reset_index(drop=True, inplace=True)
            combined_results.insert(0, 'Rank', [f'Rank_{i+1}' for i in range(len(combined_results))])

        output_path = ROOT_DIR / f'output_dir/train_results/{self.project}/{self.experiment}/{self.experiment_type}/{run}/{self.experiment_type}_{self.project}_test_combined.csv'
        combined_results.to_csv(output_path, index=False)
        logging.info(f"Saved combined test results for run {run}")