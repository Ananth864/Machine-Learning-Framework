from matplotlib import pyplot
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer,SimpleImputer
from sklearn.feature_selection import f_regression,f_classif, VarianceThreshold
from collinearity import SelectNonCollinear
from boruta import BorutaPy
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from pyspark.sql.types import NumericType, StringType
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col, when
import logging
import numpy as np 
class Dataprep:
    def __init__(self, target, type, flag_select):
        """used to initialize the dataprep
            :param target: the target feature during training
            :param type: either 'classification' or 'regression'
            :param flag_select: determines whether to perform feature selection or not
        """
        self.target = target
        self.type = type
        self.flag_select = flag_select

    def Feature_selection(self, df, col_set, model_name, cutoff, top_n, mul_col_thr, variance_threshold, threshold, method):
        """Performs feature selection on the final feature and saves the new feature set in same location. Data 
           Scientists can choose between 3 models and three feature selection methods

            :param df: The df on which feature importance is decided 
            :param col_set: The current column set passed before feature selection 
            :param model_name: The model used to determine feature importance ("dt","rf","xgb")
            :param cutoff: The cutoff importance to include column when method = "cutoff"
            :param top_n: The top_n features to include column when method = "top"
            :param threshold: Select top columns of which the cumalative feature importance is less than 
                      or equal to threshold when method = "coverage" 
            :param method: method of feature selection to be done ("top","coverage","cutoff")
            return: new_col set
        """
        if (self.flag_select == False):
            return col_set
        
        logging.info(f"Initiated Feature Selection using {method} method")

        regression_models = {"dt": DecisionTreeRegressor(), "rf": RandomForestRegressor(), "xgb": XGBRegressor(),}
        classification_models = {"dt": DecisionTreeClassifier(), "rf": RandomForestClassifier(), "xgb": XGBClassifier()}
        set_no = 0
        new_set = []
        for set in col_set:
            print("Set", set_no, "Feature importance")
            if(self.type == 'classification'):
                model = classification_models[model_name]
            else:
                model = regression_models[model_name]
            y = df[self.target]
            X = df[set]


            if(self.type == "classification"):
                selector = SelectNonCollinear(correlation_threshold=mul_col_thr,scoring=f_classif)
            else:
                selector = SelectNonCollinear(correlation_threshold=mul_col_thr,scoring=f_regression)
            selector.fit(np.array(X),np.array(y))
            mask = selector.get_support()
            kept_columns = [col for col, keep in zip(list(X), mask) if keep]
            X = df[kept_columns]
            #Removing low variance
            variance_selector = VarianceThreshold(threshold=variance_threshold)
            variance_selector.fit(X)
            mask = variance_selector.get_support(indices=False)
            selected_features = X.columns[mask]
            X = X[selected_features]

            logging.info("Retrieving Feature Importances")
            imp = []
            col_list = []
            feat_dict = {}
            if method != "boruta":
                model.fit(X,y)
                importance = model.feature_importances_
                for idx, val in enumerate(importance):
                    print('Feature:', X.columns[idx], 'Score:', round(val*100,2),'%')
                    feat_dict[X.columns[idx]] = val
                sorted_dict = dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))
                keys = list(sorted_dict.keys())
                values = list(sorted_dict.values())
            logging.info("Applying Selection method")
            if method == "boruta":
                if(self.type == 'classification'):
                    forest = RandomForestClassifier()
                else:
                    forest = RandomForestRegressor()
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, random_state=1)
                feat_selector.fit(np.array(X), np.array(y))
                mask = feat_selector.support_
                col_list = [col for col, keep in zip(list(X), mask) if keep]
                imp = feat_selector.ranking_
            elif method == "top":
                col_list = keys[:min(len(keys),top_n)]
                imp = values[:min(len(keys),top_n)]
            elif method == "coverage":
                cum_imp = 0
                for k, v in sorted_dict.items():
                    if cum_imp <= threshold:
                        col_list.append(k)
                        imp.append(v)
                        cum_imp += v
                    else:
                        break
            elif method == "cutoff":
                for k, v in sorted_dict.items():
                    if v >= cutoff:
                        col_list.append(k)
                        imp.append(v)
                    else:
                        break
            else:
                return col_set
            set_no = set_no+1
            print("\n\n")
            new_set.append(col_list)
        logging.info(f"Completed Feature Selection using {method} method\n\n")
        return new_set

    
    def data_imputation(self, spark, df, df_test, target, strategy):
        """
        Performs data imputation on missing data using specified strategy in PySpark.
        
        :param spark: SparkSession
        :param df: The DataFrame on which imputation strategy is learned and performed
        :param df_test: Test DataFrame to apply the same imputation strategy
        :param target: Target column name to exclude from imputation
        :param strategy: The imputation strategy to perform (mean, median, most_frequent)
        :return: Processed train and test DataFrames, list of categorical columns
        """
        from pyspark.sql.types import NumericType, StringType
        from pyspark.ml.feature import Imputer
        from pyspark.sql.functions import col, when, lit
        from pyspark.sql.types import DoubleType

        print("Debug: Input DataFrames")
        print("Train DataFrame Columns:", df.columns)
        print("Test DataFrame Columns:", df_test.columns if df_test is not None else "No Test DataFrame")

        valid_strategies = ['mean', 'median', 'most_frequent']
        if strategy not in valid_strategies:
            print(f"Invalid Imputation Strategy entered, defaulting to mean.")
            strategy = 'mean'
        
        # Identify numeric and non-numeric columns
        columns = [col for col in df.columns if col != target]
        
        # Identify numeric columns
        num_cols = [
            column.name for column in df.schema.fields 
            if column.name in columns and isinstance(column.dataType, NumericType)
        ]
        
        # Identify non-numeric (categorical) columns
        non_num_cols = [
            column.name for column in df.schema.fields 
            if column.name in columns and isinstance(column.dataType, StringType)
        ]

        print("Debug: Column Types")
        print("Numeric Columns:", num_cols)
        print("Non-Numeric Columns:", non_num_cols)
        
        # Replace infinity with null
        df = df.replace([float('inf'), float('-inf')], None)
        if df_test is not None:
            df_test = df_test.replace([float('inf'), float('-inf')], None)
        
        # Numeric column imputation
        if num_cols:
            # Only cast columns that exist in both train and test DataFrames
            if df_test is not None:
                common_num_cols = [col for col in num_cols if col in df_test.columns]
            else:
                common_num_cols = num_cols

            print("Debug: Common Numeric Columns:", common_num_cols)
            
            for num_col in common_num_cols:
                df = df.withColumn(num_col, col(num_col).cast(DoubleType()))
                if df_test is not None:
                    df_test = df_test.withColumn(num_col, col(num_col).cast(DoubleType()))
            
            imputer_strategy_map = {
                'mean': 'mean',
                'median': 'median',
                'most_frequent': 'mode'
            }
            
            # Prepare imputer with only common columns
            try:
                imputer = Imputer(
                    inputCols=common_num_cols, 
                    outputCols=common_num_cols, 
                    strategy=imputer_strategy_map[strategy]
                )
                
                # Fit and transform train data
                df = imputer.fit(df).transform(df)
                
                # Transform test data if provided
                if df_test is not None:
                    df_test = imputer.transform(df_test)
            
            except Exception as e:
                print(f"Imputer Error: {e}")
                # Fallback to manual imputation if Imputer fails
                for num_col in common_num_cols:
                    if strategy == 'mean':
                        mean_val = df.agg({num_col: "mean"}).collect()[0][0]
                        df = df.fillna({num_col: mean_val})
                        if df_test is not None:
                            df_test = df_test.fillna({num_col: mean_val})
                    elif strategy == 'median':
                        median_val = df.approxQuantile(num_col, [0.5], 0.0)[0]
                        df = df.fillna({num_col: median_val})
                        if df_test is not None:
                            df_test = df_test.fillna({num_col: median_val})
        
        # Categorical column imputation (mode/most frequent)
        if non_num_cols:
            # Only impute categorical columns that exist in both train and test DataFrames
            if df_test is not None:
                common_cat_cols = [col for col in non_num_cols if col in df_test.columns]
            else:
                common_cat_cols = non_num_cols

            print("Debug: Common Categorical Columns:", common_cat_cols)
            
            for col_name in common_cat_cols:
                # Find the most frequent value
                most_frequent = df.groupBy(col_name).count().orderBy('count', ascending=False).first()[col_name]
                
                # Replace null values with the most frequent value
                df = df.withColumn(col_name, 
                    when(col(col_name).isNull(), most_frequent)
                    .otherwise(col(col_name))
                )
                
                # Do the same for test data if provided
                if df_test is not None:
                    df_test = df_test.withColumn(col_name, 
                        when(col(col_name).isNull(), most_frequent)
                        .otherwise(col(col_name))
                    )

        return df, df_test, common_cat_cols