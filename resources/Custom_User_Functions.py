import numpy as np
import pandas as pd
import utils
from utils import *
import Project_Config_Parser
from Project_Config_Parser import *
import time
import pandas as pd
import numpy as np
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import pyspark as p
from pyspark.sql import SparkSession,SQLContext, HiveContext, Row, window
from pyspark import SparkConf

from dateutil.relativedelta import *
from pyspark.sql import functions as F
import warnings
warnings.filterwarnings("ignore")

import pickle
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.window import Window

from pyspark.sql.types import *

from pyspark.sql.session import SparkSession

ROOT_DIR = Path("Project_Config_Parser.py").resolve().parent.parent
RESOURCE_DIR = str(ROOT_DIR) + "/" + "resources"
conf = json_parse(RESOURCE_DIR+'/Project_Configs.json')

# Example custom user defined metric function
# def churnScorer(y_actual, y_pred): #Custom metric
#     cm = met.confusion_matrix(y_actual, y_pred).ravel().tolist()
#     cost = 60*cm[2]+ 0*cm[0] + 10*cm[1] + 10*cm[3]
#     return cost

"""
Example Function to perform data validation 

def data_validation(df,df_test):
    use pandera to perform data validation
    e.g.
    schema = pa.DataFrameSchema({
        "column1": pa.Column(int, checks=pa.Check.le(10)),
        "column2": pa.Column(float, checks=pa.Check.lt(-1.2)),
        "column3": pa.Column(str, checks=[
            pa.Check.str_startswith("value_"),
            # define custom checks as functions that take a series as input and
            # outputs a boolean or boolean Series
            pa.Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
        ]),
    })
    validated_df = schema(df)
    validated_df = schema(df)
"""
def import_remote_data(train_db, train_table, test_db, test_table):
    """User defined function that takes in train and test db and table information so that user 
        can write function to retrieve it from database and return the train and test tables
        train_db: The train table database
        train_table: the train table
        test_db: the test table database
        test_table: the test table table
    """
    return train_table, test_table

# def knn_missing_imputer_train(all_df,n_neighbors): #Custom imputers
#     pass


def custom_feat(project, experiment, df_train, df_test): #Custom Feature Engineering function 
    """user defined function to create custom features
        :project: project name 
        :experiment: experiment name
        :df_train: Pyspark dataframe containing training data
        :df_test: Pyspark dataframe containing test data
        :return: Pandas Dataframes df_train, df_test
    """
    return df_train, df_test
       
