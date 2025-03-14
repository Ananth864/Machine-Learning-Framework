"""
This script contains all the functions to read the config file.
"""
WORK_DB="WORK_DB" #to be changed later
from pathlib import PosixPath, Path
from datetime import datetime
import logging
import logging.config
from copy import deepcopy
import os
from datetime import datetime
import utils
from utils import *
ROOT_DIR = Path("Project_Config_Parser.py").resolve().parent.parent
RESOURCE_DIR = str(ROOT_DIR) + "/" + "resources"
format = "%Y%m%d"

class ProjectConfigParser:
    """Parse json config to model configurations. This contains all validation and default value processing.
    """
    def __init__(self, json_config: dict):
        """Initializing class
            :json_config: model config dictionary
            :return: None
        """
        self.output_dir: PosixPath = self.get_output_dir()
        self._json_config = json_config
        self.current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._logging_config = json_parse("logging.json")
        self.log_file = self.get_log_file()
        self.set_up_logging()
        self.project_name = self.get_project_name()
        self.experiment_name = self.get_experiment_name()
        self.target = self.get_target()
        self.project_config = self.get_project_config()
        self.data = self.get_data()
        self.experiment_type = self.get_experiment_type()
     
    def get_project_name(self):
        """Extracting project name from execution arguements
            :return: project string
        """
        try:
            project = sys.argv[1]
            if(project == ""):
                raise KeyError
            else:
                return project
        except KeyError:
                logging.warning(f"project arguement needs to be specified")
                raise ValidationError
        
    def get_experiment_name(self):
        """Extracting experiment name from execution arguements
            :return: experiment string
        """
        try:
            experiment = sys.argv[2]
            if(experiment == ""):
                raise KeyError
            else:
                return experiment
        except KeyError:
                logging.warning(f"experiment arguement needs to be specified")
                raise ValidationError
    
    def get_target(self):
        """Extracting target key from model project config
            :return: target string
        """
        try:
            target = self._json_config[self.project_name]['target']
            if(target == ""):
                raise KeyError
            else:
                return target
        except KeyError:
                logging.warning(f"target needs to be specified")
                raise ValidationError

    def get_experiment_type(self):
        """Extracting experiment type from project config
            :return: experiment type string
        """
        try:
            experiment_type = self._json_config[self.project_name]['experiment_type']
            if(experiment_type not in ["regression", "classification"]):
                raise ValidationError
            else:
                return experiment_type
        except ValidationError:
                logging.warning(f"experiment_type needs either regression or classification")
                raise ValidationError
    
    def get_project_config(self):
        """Extracting the dictionary of given project from project config
            :return: dictionary of given project
        """
        try:
            project_config = self._json_config[self.get_project_name()]
            if(project_config == ""):
                raise KeyError
            else:
                return project_config
        except KeyError:
                logging.warning(f"project config needs to be specified")
                raise ValidationError
    
    def get_data(self):
        """Extracting data key from project config
            :return: data dictionary
        """
        try:
            data = self._json_config[self.project_name]['data']
            if(data == ""):
                raise KeyError
            else:
                return data
        except KeyError:
                logging.warning(f"data needs to be specified")
                raise ValidationError 
    
    def get_output_dir(self):
        """Checking for output_dir directory which contains logging and final results and making it if not present
            :return: output directory address
        """
        if not Path(str(ROOT_DIR) + f'/output_dir').is_dir():
            os.mkdir(str(ROOT_DIR) + f'/output_dir')
        if not Path(str(ROOT_DIR) + f'/output_dir/logging').is_dir():
            os.mkdir(str(ROOT_DIR) + f'/output_dir/logging')
        OUTPUT_DIR = str(ROOT_DIR) + "/output_dir/logging"
        return OUTPUT_DIR
        
    def get_log_file(self):
        """Making a log file inside output directory
            :return: Path of log file
        """
        self.output_dir = Path(self.get_output_dir())
        log_file = self.output_dir / f'log_{self.current_datetime}.log'
        return log_file

    def set_up_logging(self):
        """Setting up logging
            :return: None
        """
        self._logging_config["handlers"]["file"]["filename"] = str(self.log_file)
        logging.captureWarnings(True)
        logging.config.dictConfig(self._logging_config)
        logging.info(f"Log file is saved at {self.log_file}")
    
    def validate_dataprep(self):
        feat_select = self._json_config[self.project_name]["feature_selection"]
        threshold = feat_select['threshold']
        cutoff = feat_select['cut-off']
        top_n = feat_select['top_n']
        method = feat_select["method"]
        include = feat_select["include"]
        mul_col_thr = feat_select["mul_col_thr"]
        variance_threshold = feat_select["variance_threshold"]
        if(method not in ["top","coverage","cut-off","boruta"]):
            logging.warning(f"Invalid  method for Feature Selection, skipping feature selection")
            include = 0
            few = input(f"Invalid  method for Feature Selection, skipping feature selection, press enter to continue")
        else:
            if((type(threshold) == str or threshold > 1 or threshold < 0) and method == "coverage"):
                threshold = 0.99
                logging.warning(f"Invalid threshold value for Feature Selection, defaulting to 0.99")
                few = input(f"Invalid threshold value for Feature Selection, defaulting to 0.99, press enter to continue")
            
            if((type(cutoff) == str or cutoff > 1 or cutoff < 0 ) and method == "cutoff"):
                cutoff = 0.01
                logging.warning(f"Invalid cutoff value for Feature Selection, defaulting to 0.01")
                few = input(f"Invalid cutoff value for Feature Selection, defaulting to 0.01, press enter to continue")
            
            if((type(top_n) != int or top_n < 1) and method == "top"):
                top_n = 10
                logging.warning(f"Invalid top_n value for Feature Selection, defaulting to 10")
                few = input(f"Invalid top_n value for Feature Selection, defaulting to 10, press enter to continue")
            
            if((type(mul_col_thr) == str or mul_col_thr > 1 or mul_col_thr < 0)):
                mul_col_thr = 0.9
                logging.warning(f"Invalid mul_col_thr value for Feature Selection, defaulting to 0.40")
                few = input(f"Invalid mul_col_thr value for Feature Selection, defaulting to 0.40, press enter to continue")

            if((type(variance_threshold) == str or variance_threshold > 1 or variance_threshold < 0)):
                variance_threshold = 0.1
                logging.warning(f"Invalid variance_threshold value for Feature Selection, defaulting to 0.40")
                few = input(f"Invalid variance_threshold value for Feature Selection, defaulting to 0.40, press enter to continue")
        

        return threshold, cutoff, top_n, include, mul_col_thr, variance_threshold
    
    def validate_training(self):
        project_config = self.get_project_config()
        parameters = project_config["parameters"]
        if type(parameters) != dict:
            logging.error(f"parameters needs to be a dictionary")
            raise ValidationError
        for key in parameters:
            if type(parameters[key]) != dict:
                logging.error(f"{key} needs to  be dictionary")
                raise ValidationError
        fold = project_config["fold"]
        if type(fold) != int:
            logging.error(f"fold needs to be an integer")
            raise ValidationError
        search = {"scikit-learn":["random", "grid"],"scikit-optimize":["bayesian"],"tune-sklearn":["random","grid","bayesian","hyperopt","optuna","bohb"],"optuna":["random","tpe"]}
        search_library = project_config["search_library"]
        if search_library not in search:
            logging.error(f"Invalid search_library. look in documentation for valid libraries")
            raise ValidationError
        search_algorithm = project_config["search_algorithm"]
        if search_algorithm not in search[search_library]:
            logging.error(f"Invalid search_algorithm. look in documentation for valid search algorithm for {search_library}")
            raise ValidationError
        custom_metrics = project_config["custom_metrics"]
        if type(custom_metrics) != list:
            logging.error(f"custom_metrics needs to be a list")
            raise ValidationError
        for metric in custom_metrics:
            if type(metric) != str:
                logging.error(f"metrics in custom_metrics need to be strings")
                raise ValidationError
            
        project = self.get_project_name()
        experiment = self.get_experiment_name()  
        experiments = project_config["experiments"]
        if type(experiments) != dict:
            logging.error(f"experiments needs to be a dictionary")
            raise ValidationError
        for exp in experiments:
            if type(experiments[exp]) != dict:
                logging.error(f"experiment needs to be a dictionary")
                raise ValidationError
        if experiment not in experiments:
            logging.error(f"{experiment} is not present in experiments of {project}")
            raise ValidationError
        else:
            train_config = experiments[experiment]
            name = experiment
        keys = ["models", "n_iter", "optimize"]
        for key in keys:
            if key not in train_config:
                logging.error(f"{key} needs to be present in dictionary {name}")
                raise ValidationError
        algorithms = ["lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac", "tr", "huber", "kr", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp","xgboost","lightgbm","catboost"]
        models = train_config["models"]
        if type(train_config["models"]) != str and type(train_config["models"]) != list:
            logging.error(f"Algorithms need to be string or list of strings")
            raise ValidationError
        elif type(models) == list:
            if models != []:
                for a in models:
                    if a not in algorithms:
                        logging.error(f"{a} is invalid. check documentation for supported algorithms")
                        raise ValidationError
        else:
            if models != "":
                if models not in algorithms:
                    logging.error(f"{models} is invalid. check documentation for supported algorithms")
                    raise ValidationError
        n_iter = train_config["n_iter"]
        if type(n_iter) != int:
            logging.error(f"n_iter needs to be an integer")
            raise ValidationError
        optimize = train_config["optimize"]
        if type(optimize) != str:
            logging.error(f"optimize needs to be a string")
            raise ValidationError
    
    def validate_ensemble(self):
        project_config = self.get_project_config()
        if type(project_config["ensemble_scoring"]) != dict:
            logging.error(f"ensemble needs to be dictionary")
            raise ValidationError
        keys = ['flag', 'method']
        ensemble = project_config["ensemble_scoring"]
        for key in keys:
            if key not in ensemble:
                logging.error(f"{key} needs to be present in ensemble")
                raise ValidationError

        if type(ensemble["method"]) != dict:
            logging.error(f"method needs to be dictionary")
            raise ValidationError
        keys = ['name', 'fn']
        for key in keys:
            if key not in ensemble["method"]:
                logging.error(f"{key} needs to be present in ensemble")
                raise ValidationError
        if type(ensemble["flag"]) != int:
            logging.error(f"flag needs to be an integer")
            raise ValidationError
        if type(ensemble["method"]["name"]) != str:
            logging.error(f"name of the method needs to be string")
            raise ValidationError
        if type(ensemble["method"]["fn"]) != str:
            logging.error(f"name of the method needs to be string")
            raise ValidationError

import sys
import logging
class MyHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)
        fmt = '%(asctime)s %(filename)18s %(funcName)s %(lineno)d %(levelname)8s: %(message)s'
        fmt_date = '%Y%m%d %T%Z'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)

