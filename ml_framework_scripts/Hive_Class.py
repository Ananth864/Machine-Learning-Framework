import pyspark
import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext, HiveContext
import logging

import utils
from utils import *
import Data_prep
from Data_prep import *
import Feature_Engineering
from Feature_Engineering import *
import Train_Class
from Train_Class import *
import Test_Class
from Test_Class import *
conf = json_parse(RESOURCE_DIR+'/Project_Configs.json')
experiment_mode = conf["experiment_mode"]
if experiment_mode == "training":
    import Custom_User_Functions
    from Custom_User_Functions import *
else:
    import Custom_User_Functions_Score
    from Custom_User_Functions_Score import *


class Data_Import:
    """Initializes sparkSession for hive and performs custom sql queries and post processing
    """
    def __init__(self,spark):
        """Initializing class
            :spark: spark session created
            :return: None
        """
        self.spark = spark

    def custom(self, year, month, train_db, train_table, test_db, test_table, spark):
        """Calls the custom_hive_table from custom class and returns table
            :year: training/scoring year
            :month: training/scoring month or 'yearly' in case of yearly data
            :train_db: training/scoring database
            :train_table: training/scoring table
            :test_db: testing database/None
            :test_table: testing table/None
            :spark: spark session created when initializing the class
            :return: Pandas dataframe resulting from the queries and post processing
        """
        logging.info("Performing custom hive operations\n\n")
        if test_db != None:
            return custom_hive_table(year, month, train_db, train_table, test_db, test_table, spark)
        else:
            return custom_score_hive_table(year, month, train_db, train_table, spark)

#Creating an Instance of Hive class

props = [
    ('spark.yarn.dist.archives', envzip + '#pyenv'),
    ("spark.cores.max",120),
    ("spark.executor.memory","25g"),
    ("spark.executor.instances", 8),
    ("spark.driver.memory", "40g"),
    ("spark.driver.maxResultSize", "25g"),
    ("spark.shuffle.service.enabled", "true"),
    ("spark.dynamicAllocation.enabled", "false"),
    ("spark.dynamicAllocation.minExecutors",10),
    ("spark.dynamicAllocation.maxExecutors",50),
    ("spark.broadcast.compress", "true"),
    ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
    ("spark.executor.memoryOverhead", 5000),
    ("spark.network.timeout", "600s"),
    ("spark.shuffle.registration.timeout", "60000"),
    ("spark.executor.extrajavaoptions", "-XX:+PrintGCDetails"),
    ('spark.yarn.queue', 'root.dtl_ops_yarn'),
    ("spark.sql.codegen.wholeStage", "false"),
    ("spark.sql.autoBroadcastJoinThreshold", -1)
    ]
props = [
            ('spark.dynamicAllocation.minExecutors', "20"),
            ('spark.dynamicAllocation.maxExecutors', "50"),
            ('spark.executor.cores', "8"),
            ('spark.executor.memory', "50g"),
            ('spark.driver.memory', '50g'),
            ('spark.akka.frameSize', "500")]
conf = pyspark.SparkConf().setAll(props)
spark = SparkSession.builder.enableHiveSupport().appName("ValidationQueries").config(conf=conf).config("hive.exec.dynamic.partition",True)\
.config("hive.exec.dynamic.partition.mode", "nonstrict").getOrCreate()
h = Hive(spark)