from pathlib import PosixPath, Path
import json
import logging
import pandas as pd
pd.set_option('use_inf_as_na', True)
from numbers import Number
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
#from pyspark.ml.functions import vector_to_array
import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.functions import col,lit
from pyspark.sql.functions import log,sqrt
import os
import sys
import time
import json
from copy import copy, deepcopy
import pyspark.sql.functions as F
import pyspark.sql.types as T
import utils
from utils import *


ROOT_DIR = Path("Feature_Engineering.py").resolve().parent.parent
RESOURCE_DIR = str(ROOT_DIR) + '/resources/encodings'  
DEFAULT_MAX_DUMMY_FEATURE_COUNT = 10
DEFAULT_THRESHOLD = 1

class FeatureEngineering:
    def __init__(self, spark, feature_config, run_folder,target_col,measure,month,year):
        self.spark = spark
        self.feature_config = feature_config
        self.target_col = target_col
        self.measure = measure
        self.month = month
        self.year = year
        self.run_folder = run_folder


    def process_minority_group_categorical_col(self, df, categorical_feature, threshold, max_dummy_feat_cnt):
        """stores an array containing all the categorical variable that are not "other" in json file categorical_feature + '.json' JSON file 
             inside the resources folder.
            :param df: Input dataframe where the records are filtered to have bad claims only. (target=1)
            :param caterogical_feature: caterogical_feature that is to be minority grouped
            :param threshold: The cut-off value of count of how many times a distinct value appears.
            :param max_dummy_feat_cnt: If more than "max_dummy_feat_cnt" of distinct values satisfies the threshold then top
                    "max_dummy_feat_cnt" values based on their counts are selected
            :return: None
        """
        import pyspark.sql.functions as f
        if threshold == None:
            th = DEFAULT_THRESHOLD
        else :
            th = threshold
        if max_dummy_feat_cnt == None or max_dummy_feat_cnt<0:
            mfc = DEFAULT_MAX_DUMMY_FEATURE_COUNT
        else :
            mfc = max_dummy_feat_cnt
        if categorical_feature not in df.columns:
            logging.error(f'Column {categorical_feature} is not present in df')
            raise ValidationError
        data_type = dict(df.dtypes)[categorical_feature]
        if data_type != 'string':
            logging.error(f'Column {categorical_feature} is not categorical')
            raise ValidationError
        try:
            ws = Window.partitionBy(f.col(categorical_feature))
            df_temp = df.withColumn('count', f.count(categorical_feature).over(ws))
            df_temp = df_temp.withColumn('categorical_feature_new', f.when(f.count(categorical_feature).over(ws)<th, 'other').otherwise(f.col(categorical_feature))).toPandas()
            value = df_temp.loc[df_temp['categorical_feature_new'] != 'other']
            value = value[[categorical_feature, 'count']]
            value = value.replace(r'\s*([^\s]+)\s*', r'\1', regex=True)
            value.sort_values(by = ['count'], inplace=True, ascending=False)
            value = list(value[categorical_feature].unique()[:mfc])
            f = str(self.run_folder) + '/encodings/'+ f'{categorical_feature}_minority_encoded.json' 
            file = open(f, "w+")
            file.write(json.dumps(value))
            file.close()
        except Exception as e:
            logging.error("Minority Encoding failed due to error - {}".format(e), exc_info=True)
        

    def encode_categorical_feature(self, df, df_test, encoding, categorical_feature):
        """Encodes categorical features using either minority grouping or one-hot encoding"""
        import pyspark.sql.functions as f
        new_cols = []
        
        if encoding == "minority-group":
            f2 = str(self.run_folder) + '/encodings/'+ f'{categorical_feature}_minority_encoded.json' 
            with open(f2) as f1:
                dic = json.load(f1)
            words_expr = f.array(*[f.lit(w) for w in dic])
            df = df.withColumn('is_other', 
                f.arrays_overlap(f.split(f.regexp_replace(f.col(categorical_feature), " ", ""), " "), words_expr))
            df = df.withColumn(categorical_feature, 
                f.when(f.col('is_other'), f.col(categorical_feature)).otherwise('other')).drop('is_other')
                
            if df_test is not None and categorical_feature in df_test.columns:
                df_test = df_test.withColumn('is_other',
                    f.arrays_overlap(f.split(f.regexp_replace(f.col(categorical_feature), " ", ""), " "), words_expr))
                df_test = df_test.withColumn(categorical_feature, 
                    f.when(f.col('is_other'), f.col(categorical_feature)).otherwise('other')).drop('is_other')
            
        elif encoding == "one-hot":
            try:
                # Get distinct values and their counts
                value_counts = df.groupBy(categorical_feature).count().orderBy('count', ascending=False)
                distinct_values = [row[0] for row in value_counts.collect() if row[0] is not None]
                
                # Create columns directly without using StringIndexer/OneHotEncoder
                for value in distinct_values:
                    col_name = f"{categorical_feature}_{value}"
                    df = df.withColumn(col_name, 
                        f.when(f.col(categorical_feature) == value, 1.0).otherwise(0.0))
                    new_cols.append(col_name)
                    
                    if df_test is not None:
                        df_test = df_test.withColumn(col_name, 
                            f.when(f.col(categorical_feature) == value, 1.0).otherwise(0.0))
                
                # Save mapping for reference
                mapping_df = pd.DataFrame({'value': distinct_values})
                mapping_df.to_csv(str(self.run_folder) + f'/encodings/{categorical_feature}_OneHot_Mappings.csv', index=False)
                
                # Drop original column
                df = df.drop(categorical_feature)
                if df_test is not None:
                    df_test = df_test.drop(categorical_feature)
                    
            except Exception as e:
                logging.error(f"Error during one-hot encoding of {categorical_feature}: {str(e)}")
                raise
                
        logging.info(f"Completed {encoding} encoding on {categorical_feature}")
        return df, df_test, new_cols
            
        
    def handle_categorical_feature(self, df=None, df_test=None, feature_name = None, encoding = None, bad_claim_threshold = None, max_dummy_feat_cnt = None):
        """Handles/creates categotical feature based on the encoding_type. A sample encoding_type would look like as below.
                "encoding": "minority-group"    
                or "encoding": "one-hot"
                Max dummy features also implemented for one-hot encoding and other features kept as others
            :param spark: spark session
            :param df: Pyspark Dataframe that contains key columns and the  categorical features that is to be processed.
            :param df_test: Test Pyspark Dataframe that contains key columns and the  categorical features that is to be processed.
            :param feature_name: The feature name that is to be handled
            :param encoding: string that tells encoding to be performed. e.g. "minority-group"
            :param bad_claim_threshold: The "associated count" cut-off that determines whether a dummy variable should be created for a distinct value of
                                        categorical feature. here "associated count" is how many times that distinct value present when the
                                        target column is '1' which is bad
            :param max_dummy_feat_cnt: Tells the maximum number of dummy variables to be created.
            :return: processed df, df_test
        """
        logging.info(f"Performing {encoding} encoding on {feature_name}")
        import pyspark.sql.functions as f
        if encoding == "minority-group":
            if self.target_col != None:
                df_bad = df.filter(f.col(self.target_col) == 1)
            #process MinorityGrouping
            else:
                df_bad = df

            self.process_minority_group_categorical_col(df_bad, feature_name, bad_claim_threshold, max_dummy_feat_cnt)
        #encode OnehotEncoding
        return self.encode_categorical_feature(df, df_test, encoding, feature_name)


    def handle_numerical_feature(self, df, df_test, calc_feature, calculation_str, fn):
        """Handles/creates numerical feature based on the calculation config. A sample calculation config would look like as below.
                "ratio":{"calculation" : "num/denum", "fn" : "log"}
                The calculation string will be parsed as operator_to_apply = '/', left_operand='num', right_operand='denum' and finally  fn = 'log' applied
                Based on these operator and operands the calculation will be performed
            :param df: Pyspark Dataframe that contains the features that are to processed for the given calculation.
            :param df_test: Pyspark Dataframe that contains the features that are to processed for the given calculation.
            :param calc_feature: Feature to be created after applying the caluclation
            :param calculation_str: String that contains calculation to be performed
            :param fn: String that contains non-linear fn to be performed (log, sqrt)
            :return: processed df, df_test
        """
        logging.info(f"Performing numerical function {fn} on {calculation_str} to make {calc_feature}")
        str1 = calculation_str
        new_cols = []
        try:
            single = 1
            for i in range(0, len(str1)):
                if str1[i] in op_arr:
                    ind = i
                    single = 0 
                    break
            if (single):
                if(str1 not in df.columns):
                    raise KeyError
                if(fn == "log"):
                    df = df.withColumn(calc_feature, log(calculation_str))
                if(fn == "sqrt"):
                    df = df.withColumn(calc_feature, sqrt(calculation_str))
            else:   
                if((str1[0:ind] in df.columns) and (str1[i+1:len(str1)] in df.columns)):
                    val = operator(df[str1[0:ind]], str1[ind], df[str1[i+1:len(str1)]])
                    if(fn == "log"):
                        df = df.withColumn(calc_feature, log(val))
                    elif(fn == "sqrt"):
                        df = df.withColumn(calc_feature, sqrt(val))
                    else:
                        df = df.withColumn(calc_feature,val)
                else:
                    raise KeyError
        
        except KeyError:
            #logging.warning(f"error")
            #raise ValidationError
            logging.error("string format or column name incorrect in train data\n")
            pass
        
        if (df_test is None):
            return df,df_test, calc_feature
        
        try:
            single = 1
            for i in range(0, len(str1)):
                if str1[i] in op_arr:
                    ind = i
                    single = 0 
                    break
            if (single):
                if(str1 not in df_test.columns):
                    raise KeyError
                if(fn == "log"):
                    df_test = df_test.withColumn(calc_feature, log(calculation_str))
                if(fn == "sqrt"):
                    df_test = df_test.withColumn(calc_feature, sqrt(calculation_str))
            else:   
                if((str1[0:ind] in df_test.columns) and (str1[i+1:len(str1)] in df_test.columns)):
                    val = operator(df_test[str1[0:ind]], str1[ind], df_test[str1[i+1:len(str1)]])
                    if(fn == "log"):
                        df_test = df_test.withColumn(calc_feature, log(val))
                    elif(fn == "sqrt"):
                        df_test = df_test.withColumn(calc_feature, sqrt(val))
                    else:
                        df_test = df_test.withColumn(calc_feature,val)
                else:
                    raise KeyError
        
        except KeyError:
            #logging.warning(f"error")
            #raise ValidationError
            logging.error("string format or column name incorrect in test data\n")
            pass
            
        logging.info(f"Performing numerical encoding to make {calc_feature}")
        return df,df_test,calc_feature


    def handle_each_feature(self, df, df_test, feature_name, technique):
        """It peroforms specified "technique" on the given input "feature_name" and retruns the processed df.

            :param feature_name: The feature name that is to be handled
            :param technique: disctionary of a technique that tells how to hanlde the feature
            :return: processed df, df_test 
        """
        if "encoding" in technique:
            if 'threshold' in technique:
                th = technique['threshold']
            else:
                th = None
            if 'max_dummy_feat_cnt' in technique:
                mfc = technique['max_dummy_feat_cnt']
            else:
                mfc = None
            return self.handle_categorical_feature(df, df_test, feature_name, technique['encoding'], th, mfc)
        elif "calculation" in technique:
            return self.handle_numerical_feature(df, df_test, feature_name, technique["calculation"], technique["fn"])
        else:
            return df, df_test, None


    def apply_feature_engineering(self, df, df_test):
        """Performs feature engineering on the specified columns mentioned in FEAT_ENG using the specified feature engineering technique.
                    A sample feature_config would look like below.
                    "feature_engineering": {
                    "ratio":{
                        "calculation" : "num/denum", 
                        "fn" : "log"
                    }, 
                    "Minority":{
                        "encoding" :"minority-group", 
                        "threshold" : 2, 
                        "max_dummy_feat_cnt" : 2
                    }
                },
            :return: processed df,df_test 
        """
        for feature_name in self.feature_config:
                logging.info(f"Initiating feature engineering...")
                df, df_test, new_cols = self.handle_each_feature(df, df_test, feature_name ,self.feature_config[feature_name])
                logging.info(f"Completed feature engineering\n\n")
        return df, df_test
    

    def cattoonehot(self, df, df_test, features, target, cat_list, run):
        """Performs automatic one-hot encoding on categorical columns with optimized memory usage"""
        logging.info("Performing One-hot encoding on categorical columns")
        cat_dict = {}
        train_cols = deepcopy(features['training_features'])
        
        # Save categorical columns for reference
        f = str(self.run_folder) + '/encodings/categorical_columns.json'
        with open(f, "w+") as file:
            json.dump(cat_list, file)

        # Process each feature set
        for i, feature_set in enumerate(train_cols):
            logging.info(f"Processing set_{i+1}")
            new_feature_set = []
            
            for col in feature_set:
                if col not in cat_list:
                    new_feature_set.append(col)
                    continue
                    
                if col in cat_dict:
                    # Use previously computed encodings
                    new_feature_set.extend(cat_dict[col])
                    continue
                    
                logging.info(f"Processing categorical column: {col}")
                start_time = time.time()
                
                try:
                    # Process entire column in one go
                    df, df_test, encoded_cols = self.handle_categorical_feature(
                        df, 
                        df_test, 
                        col, 
                        encoding='one-hot'
                    )
                    
                    if encoded_cols:
                        cat_dict[col] = encoded_cols
                        new_feature_set.extend(encoded_cols)
                        
                    logging.info(f"Encoded {col} in {time.time() - start_time:.2f} seconds")
                    
                except Exception as e:
                    logging.error(f"Error processing column {col}: {str(e)}")
                    # If encoding fails, keep original column
                    new_feature_set.append(col)
                    continue
                
            train_cols[i] = new_feature_set

        features['training_features'] = train_cols
        logging.info("Completed One-hot encoding on categorical columns")
        return df, df_test, features