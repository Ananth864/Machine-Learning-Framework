if __name__ == '__main__':
    import logging
    import os
    import sys
    import time
    from pathlib import Path

    import findspark
    import pandas as pd
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pycaret.regression import *

    # Initialize Spark and set up logging
    findspark.init()
    logging.getLogger().setLevel(logging.INFO)
    os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
    import os
    os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-1.8"  # Adjust path
    os.environ["SPARK_HOME"] = "C:/Users/ASUS/anaconda3/envs/ml_framework/lib/site-packages/pyspark"

    # Set up project directories
    ROOT_DIR = Path("main.py").resolve().parent.parent
    RESOURCE_DIR = ROOT_DIR / "resources"
    sys.path.insert(1, str(ROOT_DIR/'resources'))

    # Import local modules
    from utils import *
    from Data_prep import *
    from Feature_Engineering import *
    from Train_Class import *
    from Test_Class import *
    from Custom_User_Functions import *
    from Project_Config_Parser import *

    def initialize_project():
        """Initialize project configuration by reading and parsing project config.
    
        Returns:
            tuple: (ProjectConfigParser object, project name string, experiment name string)
        """
        conf = json_parse(RESOURCE_DIR+ '/Project_Configs.json')
        config = ProjectConfigParser(conf)
        project = config.project_name
        experiment = config.experiment_name
        
        return config, project, experiment

    def load_data(config):
        """Load train and test data from local or remote sources based on config.
    
        Args:
            config: ProjectConfigParser object containing data configuration
            
        Returns:
            tuple: (SparkSession, cached train DataFrame, cached test DataFrame)
            
        Raises:
            ValidationError: If data loading fails
        """
        spark = SparkSession.builder.appName('Practise').getOrCreate()
        
        if config.data['data_retrieval_mode'] == 'local':
            try:
                df = spark.read.csv(config.data['local']['train_table_path'], header=True, inferSchema=True)
                df_test = spark.read.csv(config.data['local']['test_table_path'], header=True, inferSchema=True)
            except:
                logging.error("Failed to read datasets from local location. Check location in ProjectConfig under 'local' dictionary")
                raise ValidationError("Failed to read datasets from local location")
        else:
            try:
                df, df_test = import_remote_data(
                    config.data['remote']['train_db'],
                    config.data['remote']['train_table'],
                    config.data['remote']['test_db'],
                    config.data['remote']['test_table']
                )
            except:
                logging.error("Failed to read datasets from remote location. Check names in ProjectConfig under 'remote' dictionary")
                raise ValidationError("Failed to read datasets from remote location")
                
        return spark, df.cache(), df_test.cache()

    def process_features(spark, df, df_test, config, project, experiment, experiment_type, run, target):
        """Process data through feature engineering pipeline including custom engineering and data prep.
    
        Args:
            spark: SparkSession
            df: Training DataFrame
            df_test: Test DataFrame 
            config: Project configuration object
            project: Project name
            experiment: Experiment name
            experiment_type: Type of experiment (regression/classification)
            run: Run identifier
            target: Target column name
            
        Returns:
            tuple: (Processed train df, processed test df, categorical columns list,
                data preparation object, feature engineering object)
        """

        rn_folder = ROOT_DIR / f'models/{project}/{experiment}/{experiment_type}/{run}'
        feat_eng = FeatureEngineering(spark, config.project_config['feature_engineering'], 
                                    str(rn_folder), None, 
                                    project, experiment_type, experiment)
        df_sm, df_sm_test = feat_eng.apply_feature_engineering(df, df_test)
        
        # Custom feature engineering
        df_sm, df_sm_test = custom_feat(project, experiment, df_sm, df_sm_test)
        
        # Data preparation
        threshold, cutoff, top_n, include, mul_col_thr, variance_threshold = config.validate_dataprep()
        data_preparation = Dataprep(target, 'regression', include)
        df_sm, df_sm_test, cat_list = data_preparation.data_imputation(
            spark, df_sm, df_sm_test, target, config.project_config['imputation_method']
        )
        
        return df_sm, df_sm_test, cat_list, data_preparation, feat_eng

    def handle_training_columns(df_sm, df_sm_test, project, experiment, experiment_type, run, target, cat_list, feat_eng):
        """Process training column sets and apply one-hot encoding.
    
        Args:
            df_sm: Training DataFrame
            df_sm_test: Test DataFrame
            project: Project name 
            experiment: Experiment name
            experiment_type: Type of experiment
            run: Run identifier
            target: Target column name
            cat_list: List of categorical columns
            feat_eng: Feature engineering object
            
        Returns:
            tuple: (Processed train df, processed test df, features dict, final feature list)
            
        Raises:
            ValidationError: If column processing fails
        """
        feature_set_final = []
        training_cols_path = str(ROOT_DIR) + '/resources/Training_Columns.json'
        with open(training_cols_path, 'r+') as f:
            try:
                features = json.load(f)
            except:
                logging.error("Please Ensure that Training_Columns.json is filled properly")
                raise ValidationError
            features["training_features"] = remove_invalid_columns(features["training_features"], df_sm)
            
            if not features["training_features"]:
                temp2 = list(df_sm.columns)
                print(temp2)
                try:
                    temp2.remove(target)
                except:
                    logging.error("Please ensure target column is in dataset")
                    raise ValidationError
                features["training_features"] = [temp2]
            # Save features
            f.seek(0)
            f.truncate()
            json.dump(features, f)
        output_path = ROOT_DIR / f'models/{project}/{experiment}/{experiment_type}/{run}/train_columns.json'
        with open(output_path, 'w') as f:
            json.dump(features, f)
        # One-hot encoding
        try:
            train_cls = list(set([item for sublist in features["training_features"] for item in sublist]))
            df1 = df_sm_test[train_cls]
        except Exception as e:
            logging.error(str(e))
            raise ValidationError
            
        df_sm, df_sm_test, features = feat_eng.cattoonehot(df_sm, df_sm_test, features, target, cat_list, run)
        
        feature_set_final = list(set(sum(features['training_features'], [])))
        
        # Save final features
        features_path = ROOT_DIR / f'models/{project}/{experiment}/{experiment_type}/{run}/features/{experiment_type}_{project}_features.json'
        with open(features_path, 'w') as f:
            json.dump(features, f)
            
        return df_sm, df_sm_test, features, feature_set_final

    def main():
        """Main execution function that runs the complete ML pipeline.
    
        Pipeline steps:
        1. Initialize project config
        2. Load and preview data
        3. Process features
        4. Handle training columns 
        5. Convert to pandas and validate
        6. Save datasets
        7. Train models
        8. Test models
        9. Clean up logs
        """
        start_time = time.time()
        
        # Initialize project configuration
        config, project, experiment = initialize_project()
        experiment_type = config.experiment_type
        run = make_folders(project, experiment, experiment_type)
        
        # Load data
        spark, df, df_test = load_data(config)
        
        # Preview data
        print("Train Dataset head():")
        df.show(5)
        print("Test Dataset head():")
        df_test.show(5)
        
        if input("Continue execution? (Enter anything for yes, 0 for no): ") == "0":
            return
            
        # Process features
        df_sm, df_sm_test, cat_list, data_preparation, feat_eng = process_features(
            spark, df, df_test, config, project, experiment, experiment_type, run, config.target
        )
        
        # Handle training columns
        df_sm, df_sm_test, features, feature_set_final = handle_training_columns(
            df_sm, df_sm_test, project, experiment, experiment_type, 
            run, config.target, cat_list, feat_eng
        )
        
        # Convert to pandas and validate target
        df_sm = df_sm.select("*").toPandas()
        df_sm_test = df_sm_test.select("*").toPandas()
        
        if config.target not in df_sm or config.target not in df_sm_test:
            logging.warning("Target column missing in training data")
            raise ValidationError
            
        # Save datasets
        save_path = ROOT_DIR / f'output_dir/train_results/{project}/{experiment}/{experiment_type}/{run}/datasets'
        df_sm.to_csv(save_path / f'{experiment_type}_{experiment}_{project}_train.csv', index=False)
        
        for i, feature_set in enumerate(features["training_features"], 1):
            df_sm[feature_set].to_csv(save_path / f'train_set_{i}_dataset.csv', index=False)
            df_sm_test[feature_set].to_csv(save_path / f'test_set_{i}_dataset.csv', index=False)
        
        # Display final datasets
        print("\nFinal train dataset head():")
        print(df_sm[feature_set_final].head())
        print("\nFinal test dataset head():")
        print(df_sm_test[feature_set_final].head())
        
        if input("\nPress enter to continue") not in ["0", "n", "no"]:
            # Copy log file and clean up
            shutil.copy(config.get_log_file(), str(ROOT_DIR / f'models/{project}/{experiment}/{experiment_type}/{run}'))
            
            # Train and test
            config.validate_training()
            training = Train(project, experiment, experiment_type, config.project_config, config, config.target)
            testing = Test(project, experiment, experiment_type, config.project_config, config.target)
            
            training.train_feat_sets(df_sm, run)
            testing.test_feat_set(df_sm_test, run)
            
            # Clean up log file
            try:
                os.remove(Path("main.py").resolve().parent / 'logs.log')
            except OSError:
                pass

    main()