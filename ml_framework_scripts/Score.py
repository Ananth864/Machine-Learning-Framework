from pathlib import Path
import sys
import logging
import test
import pandas as pd
import pickle
import json
import os
import sys
from pycaret.regression import *
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import *
import Custom_User_Functions
from Custom_User_Functions import *
import Custom_User_Functions_Score
from Custom_User_Functions_Score import *
import utils
from utils import *
ROOT_DIR = str(Path("main.py").resolve().parent.parent)
from bayes_opt import BayesianOptimization
NEW_APPROACH_MEASURE = [59,8,11,7]
class Score:
    """This class is used for performing scoring and ensembling on all the trained models and feature sets provided in model_score.json
    """
    def __init__(self, year, month, measure, measure_config, models, df_score_list, target_list):
        self.year = year
        self.month = month
        self.measure = measure
        self.measure_config = measure_config
        self.models = models
        self.df_score_list = df_score_list
        self.target_list = target_list
        """Initializing class
            :year: Scoring year
            :month: Scoring month/ 'yearly' in case of yearly model
            :measure: measure
            :measure_config: dictionary of the given measure in config file
            :models: list of all the models mentioned in model_score.json
            :df_score_list: list of pandas dataframes containing score data 
            :target_list: list of targets used for naming the predicted label
        """
    def predict(self, run):
        """predicts the label for the all dataframes present in df_score_list
            :run: Current run number
        """
        logging.info(f"Starting scoring ...")
        for i in range(0,len(self.models)):
            m = load_model(self.models[i])
            logging.info(f"Initiating scoring on m{i+1}")
            pred = predict_model(m, self.df_score_list[i])
            if self.measure_config["MEASURE_ID"] in NEW_APPROACH_MEASURE:
                pred["Label"] = pred['Label']+pred['SM_CURR']
            pred.rename(columns = {'Label':f'{self.target_list[i]}'}, inplace = True)
            pred.to_csv(ROOT_DIR + f'/output_dir/score_results/{self.measure}/{self.year}/{self.month}/{run}/'+ self.month +f'_{self.measure}_score_m{i+1}.csv',index=False)
            logging.info(f"Finished scoring on m{i+1}")
        logging.info(f"Finished scoring")
    
    def bayesian_opt(self, params_range, fn):
        """performs bayesian optimization to determine optimal weights for ensemble prediction
            :params_range: dictionary containing weight name as key and a range as value. for ex- {'w1':(0.001,1)}
            :fn: user defined custom function written in custom_score. for ex - _build_model(w1,w2)
        """
        optimization_engine = BayesianOptimization(fn, params_range, random_state=100, verbose=0)
        optimization_engine.maximize(init_points=20, n_iter=4)
        best_params = optimization_engine.max['params']
        return best_params

    def ensemble(self, run):
        """performs ensemble (weighted average) of models provided in model_score.json using method provided in config
            :run: current run number
        """
        logging.info(f"Starting Ensemble scoring ...")
        method = self.measure_config['ensemble_scoring']['method']
        if method['name'] == "":
            self.predict(run)
        elif method['name'] == 'bayesian':
            l = len(self.models)
            params_range = {}
            for i in range(0,l):
                params_range[f'w{i+1}'] = (0.001, 1)
            f = method['fn']
            fn = importvar(f, 'Custom_Feature_Engineering_score', 'function not found')
            if fn == 'function not found':
                logging.error(f'function {f} not found in Custom_Feature_Engineering_score')
                raise ValidationError
            best_params = self.bayesian_opt(params_range, fn)
            jsondata = json.dumps(best_params)
            with open(ROOT_DIR + f'/output_dir/score_results/{self.measure}/{self.year}/{self.month}/{run}/'+ self.month +f'_{self.measure}_best_ensemble_weights.json', 'w') as f1:
                f1.write(jsondata)
            sum = 0
            for i in range(0,l):
                sum += best_params[f'w{i+1}']
            m = load_model(self.models[0])
            pred = predict_model(m, self.df_score_list[0])
            if self.measure_config["MEASURE_ID"] in NEW_APPROACH_MEASURE:
                pred["Label"] = pred['Label']+pred['SM_CURR']
            ensemble_prediction = best_params[f'w{1}']*pred.Label/sum 
            for i in range(1,l):
                m = load_model(self.models[i])
                pred = predict_model(m, self.df_score_list[i])
                if self.measure_config["MEASURE_ID"] in NEW_APPROACH_MEASURE:
                    pred["Label"] = pred['Label']+pred['SM_CURR']
                ensemble_prediction += best_params[f'w{i+1}']*pred.Label/sum
            for i in range(0,l):
                self.df_score_list[i]['ensemble_prediction'] = ensemble_prediction
                self.df_score_list[i].to_csv(ROOT_DIR + f'/output_dir/score_results/{self.measure}/{self.year}/{self.month}/{run}/'+ self.month +f'_{self.measure}_ensemble_score_{i+1}.csv',index=False)
            logging.info(f"Finished Ensemble scoring")
