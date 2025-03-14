from pathlib import Path
import logging
import pandas as pd
import pickle
import json
from pycaret.regression import *
from sklearn.base import clone
from Custom_User_Functions import *
from utils import *

ROOT_DIR = Path("main.py").resolve().parent.parent

class Train:
    """Class for training ML models with support for multiple feature sets, model types, and model combinations."""
    
    def __init__(self, project, experiment, experiment_type, project_config, config, target):
        """Initialize Train class.

        Args:
            project: Project name
            experiment: Experiment name 
            experiment_type: Type of experiment (regression/classification)
            project_config: Dictionary containing project configuration
            config: ProjectConfigParser object
            target: Target column name
        """
        self.experiment = experiment
        self.experiment_type = experiment_type
        self.project = project
        self.project_config = project_config
        self.config = config
        self.target = target
        self.train_config = project_config['experiments'][experiment]
        self.fold = self._validate_fold(project_config['fold'])

    def _validate_fold(self, fold):
        """Validate fold parameter for cross validation.
        
        Args:
            fold: Number of folds for cross validation
            
        Returns:
            int: Valid fold number, defaults to 5 if invalid
        """
        if not isinstance(fold, int) or fold < 1:
            logging.info("Invalid fold value provided, defaulting to 5")
            return 5
        return fold

    def _setup_experiment(self, data):
        """Setup PyCaret experiment with given data and parameters.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            PyCaret setup object
            
        Raises: 
            ValidationError: If setup fails
        """
        setup_params = self.project_config["setup_params"]
        setup_params.update({
            "data": data,
            "target": self.target,
            "train_size": 0.99,
            "fold": self.fold,
            "preprocess": False,
            "silent": True
        })
        
        try:
            return setup(**setup_params)
        except Exception as e:
            logging.error(str(e))
            raise ValidationError

    def _add_custom_metrics(self):
        """Add custom metrics from config to PyCaret.
        
        Looks for metrics in Custom_Feature_Engineering module and sklearn.metrics.
        """
        for metric in self.project_config["custom_metrics"]:
            met = importvar(metric, 'Custom_Feature_Engineering', 'metric not found')
            if met != 'metric not found':
                add_metric(metric, metric, met)
            elif importvar(metric, 'sklearn.metrics', 'metric not found') != 'metric not found':
                add_metric(metric, metric, importvar(metric, 'sklearn.metrics', 'metric not found'))
            else:
                logging.warning(f"metric {metric} not found")

    def _create_and_tune_model(self, algo, i):
        """Create and tune a model using specified algorithm.
        
        Args:
            algo: Model algorithm name
            i: Feature set index
            
        Returns:
            Trained and tuned model object
        """
        try:
            model = create_model(algo, return_train_score=True)
        except Exception as e:
            logging.warning(e)
            algo = "huber"
            model = create_model(algo, return_train_score=True)
            input(f"Error creating {algo} model on set_{i}. Using huber instead. Press Enter to continue")

        try:
            if algo in self.project_config["parameters"]:
                return tune_model(
                    model,
                    custom_grid=self.project_config["parameters"][algo],
                    search_library=self.project_config['search_library'],
                    search_algorithm=self.project_config['search_algorithm'],
                    return_train_score=True,
                    n_iter=self.train_config["n_iter"],
                    optimize=self.train_config["optimize"]
                )
            return tune_model(
                model,
                optimize=self.train_config["optimize"],
                n_iter=self.train_config["n_iter"],
                return_train_score=True
            )
        except Exception as e:
            logging.warning(e)
            input(f"Error tuning {algo} on set_{i}. Using initial weights. Press Enter to continue")
            return model

    def _save_model_results(self, model, i, run, prefix="", model_type=""):
        """Save trained model and its results.
        
        Args:
            model: Trained model object
            i: Feature set index
            run: Run identifier
            prefix: Optional prefix for model name
            model_type: Optional model type identifier
            
        Returns:
            DataFrame containing training results
        """
        model_name = f"{prefix}{type(model).__name__}"
        train_result = pull().iloc[[self.fold*2, self.fold*2+2], :]
        train_result.insert(0, "Split", ['CV-Train', 'CV-Val'])
        train_result.insert(0, "Model", [model_name, model_name])

        model_path = ROOT_DIR / f'models/{self.project}/{self.experiment}/{self.experiment_type}/{run}/model/{self.experiment_type}_{self.project}_model_set_{i}_{model_type}{type(model).__name__}'
        save_model(model, str(model_path))
        logging.info(f"Saved {model_name} trained on set_{i}")

        results_path = ROOT_DIR / f'output_dir/train_results/{self.project}/{self.experiment}/{self.experiment_type}/{run}/{self.experiment_type}_{self.project}_set_{i}_{model_name}.csv'
        train_result.to_csv(results_path, index=False)
        logging.info(f"Saved metrics for {model_name} trained on set_{i}")

        return train_result

    def train(self, features, data, i, results, run, keys):
        """Train models on a given feature set.
        
        Args:
            features: List of feature names
            data: Training data
            i: Feature set index
            results: List to store training results
            run: Run identifier
            keys: List to store feature set keys
            
        Returns:
            List of trained models
        """
        self._setup_experiment(data)
        self._add_custom_metrics()
        
        algos = [self.train_config["models"]] if isinstance(self.train_config["models"], str) else self.train_config["models"]
        if not algos or algos == [""]:
            return []

        model_list = []
        for algo in algos:
            logging.info(f"Training started using {algo} on set_{i}")
            model = self._create_and_tune_model(algo, i)
            model_list.append(finalize_model(model))

        try:
            n_select = self.project_config['Top_N_Models']
            model_list = compare_models(model_list, n_select=n_select, sort=self.train_config["optimize"])
            if not isinstance(model_list, list):
                model_list = [model_list]
        except Exception as e:
            logging.error(str(e))
            raise ValidationError

        for model in model_list:
            create_model(model, return_train_score=True)
            result = self._save_model_results(model, i, run)
            results.append(result)
            keys.extend([f'set_{i}', f'set_{i}'])

        return model_list

    def _handle_model_combinations(self, model_list, i, run, results, keys):
        """Handle ensemble, stacking and blending of models.
        
        Args:
            model_list: List of trained models
            i: Feature set index
            run: Run identifier
            results: List to store results
            keys: List to store feature set keys
        """
        if self.project_config["Ensemble_Models"]["ensemble_flag"]:
            self._ensemble_models(model_list, i, run, results, keys)
        
        if self.project_config["Stack_Models"]["stack_flag"]:
            self._stack_models(model_list, i, run, results, keys)
            
        if self.project_config["Blend_Models"]:
            self._blend_models(model_list, i, run, results, keys)

    def _ensemble_models(self, model_list, i, run, results, keys):
        """Create ensemble models using bagging/boosting.
        
        Args:
            model_list: List of base models
            i: Feature set index 
            run: Run identifier
            results: List to store results
            keys: List to store feature set keys
            
        Raises:
            ValidationError: If ensemble creation fails
        """
        config = self.project_config["Ensemble_Models"]
        ensemble_params = {
            "method": config["method"],
            "n_estimators": config["n_estimators"],
            "return_train_score": True,
            "optimize": self.train_config["optimize"]
        }

        for model in model_list:
            try:
                ensemble = ensemble_model(estimator=model, **ensemble_params)
                result = self._save_model_results(ensemble, i, run, prefix="ensemble_", model_type="Ensemble")
                results.append(result)
                keys.extend([f'set_{i}', f'set_{i}'])
            except Exception as e:
                logging.error(str(e))
                raise ValidationError

    def _stack_models(self, model_list, i, run, results, keys):
        """Create stacked models using meta-learner.
        
        Args:
            model_list: List of base models
            i: Feature set index
            run: Run identifier
            results: List to store results
            keys: List to store feature set keys
            
        Raises:
            ValidationError: If stacking fails
        """
        config = self.project_config["Stack_Models"]
        stack_params = {
            "estimator_list": model_list,
            "return_train_score": True,
            "optimize": self.train_config["optimize"],
            "restack": config["restack"]
        }

        try:
            if config["meta_model"]:
                stack_params["meta_model"] = create_model(config["meta_model"])
        except Exception:
            logging.warning("Invalid meta model. Using Linear Regression as default.")

        try:
            stack = stack_models(**stack_params)
            result = self._save_model_results(stack, i, run, model_type="stacked")
            results.append(result)
            keys.extend([f'set_{i}', f'set_{i}'])
        except Exception as e:
            logging.error(str(e))
            raise ValidationError

    def _blend_models(self, model_list, i, run, results, keys):
        """Create blended model using averaging of predictions.
        
        Args:
            model_list: List of base models
            i: Feature set index
            run: Run identifier
            results: List to store results
            keys: List to store feature set keys
            
        Raises:
            ValidationError: If blending fails
        """
        try:
            model_copies = [clone(model) for model in model_list]
        except Exception as e:
            logging.warning(f"Model cloning failed: {str(e)}. Using originals.")
            model_copies = model_list

        try:
            blend = blend_models(
                model_copies,
                optimize=self.train_config["optimize"],
                return_train_score=True
            )
            result = self._save_model_results(blend, i, run, model_type="blended")
            results.append(result)
            keys.extend([f'set_{i}', f'set_{i}'])
        except Exception as e:
            logging.error(f"Blending failed: {str(e)}")
            raise ValidationError

    def train_feat_sets(self, training_data, run):
        """Main method to train models on all feature sets.
        
        Args:
            training_data: Complete training DataFrame
            run: Run identifier
        """
        logging.info("Starting training...")
        features_path = ROOT_DIR / f'models/{self.project}/{self.experiment}/{self.experiment_type}/{run}/features/{self.experiment_type}_{self.project}_features.json'
        
        with features_path.open('r') as f:
            training_features = json.load(f)['training_features']

        feature_mapping = {}
        combined_results = []
        keys = []

        for i, features in enumerate(training_features, 1):
            feature_mapping[f'set_{i}'] = features
            training_cols = features + [self.target]
            
            try:
                subset_data = training_data[training_cols]
            except Exception as e:
                logging.warning(str(e))
                continue

            logging.info(f"Training on set_{i}")
            results = []
            model_list = self.train(features, subset_data, i, results, run, keys)
            
            try:
                self._handle_model_combinations(model_list, i, run, results, keys)
            except Exception as e:
                logging.error(str(e))
                raise ValidationError

            if results:
                combined_results.append(pd.concat(results, ignore_index=True))
            logging.info(f"Completed training on set_{i}")

        self._save_feature_mapping(feature_mapping, run)
        self._save_combined_results(combined_results, keys, run)

    def _save_feature_mapping(self, mapping, run):
        """Save feature set mapping to files.
        
        Args:
            mapping: Dictionary mapping set indices to feature lists
            run: Run identifier
        """
        mapping_json = json.dumps(mapping)
        for path in [
            ROOT_DIR / f'output_dir/train_results/{self.project}/{self.experiment}/{self.experiment_type}/{run}/{self.experiment_type}_{self.project}_set_mapping.json',
            ROOT_DIR / f'models/{self.project}/{self.experiment}/{self.experiment_type}/{run}/features/{self.experiment_type}_{self.project}_set_mapping.json'
        ]:
            with path.open('w') as f:
                f.write(mapping_json)
        logging.info("Set mapping saved")

    def _save_combined_results(self, combined_results, keys, run):
        """Save combined training results to CSV.
        
        Args:
            combined_results: List of training result DataFrames
            keys: List of feature set keys
            run: Run identifier
        """
        if not combined_results:
            return

        results_df = pd.concat(combined_results, ignore_index=True)
        results_df.insert(0, "Set", keys)
        results_df = making_rankings_train(results_df)
        
        output_path = ROOT_DIR / f'output_dir/train_results/{self.project}/{self.experiment}/{self.experiment_type}/{run}/{self.experiment_type}_{self.project}_train_combined.csv'
        results_df.to_csv(output_path, index=False)
        logging.info("Combined metrics saved for all set-model combinations")