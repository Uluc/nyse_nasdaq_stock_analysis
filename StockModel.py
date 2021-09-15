from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.regression import DecisionTreeRegressor

from pyspark.ml.regression import GBTRegressor

from pyspark.sql.functions import col,asc

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from pyspark.sql.types import StructType,StructField, StringType, IntegerType

import pyspark.sql.functions as F

stringOutput =""

arg1 = "s3://stockanalysisbucket2.0/full_history/AAPL.csv"
df = spark.read.format('csv').options(header='true', inferSchema="true").load(arg1)
#file contains all stock symbols, which are used to load the data into dataframe
log_txt=sc.textFile("s3://stockanalysisbucket2.0/all_symbols.txt")

file = log_txt.collect()

def machineLearningModel(name):
    
    stringOutput =""

    arg1 = "s3://stockanalysisbucket2.0/full_history/"+name+".csv"
    #read data into dataframe
    df = spark.read.format('csv').options(header='true', inferSchema="true").load(arg1)

    #get dates for appending to tables later
    df2 = df.select("date")
    df2 = df2.sort(asc("date"))
    df2 = df2.withColumn("id", F.monotonically_increasing_id())
    
    #use vectorassembler to select relevant dataframe columns
    vectorAssembler = VectorAssembler(inputCols = ['open', 'volume', 'high', 'low'], outputCol = 'features')

    #run linear regression algorithm on data
    df = vectorAssembler.transform(df)
    df = df.select(['features', 'close'])
    train_df, test_df = df.randomSplit([0.7, 0.3])
    lr = LinearRegression(featuresCol = 'features', labelCol='close', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(df)
    lr_pred = lr_model.transform(test_df)
    lr_pred = lr_pred.withColumn("id", F.monotonically_increasing_id())
    lr_pred = lr_pred.join(df2, on="id")
    lr_pred.select("prediction","close","features","date")
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="close",metricName="r2")
    test_result = lr_model.evaluate(test_df)
    stringOutput += ("Linear Reg RMSE = " + str(test_result.rootMeanSquaredError))
    pred = lr_model.transform(test_df)
    pred = pred.withColumn("id", F.monotonically_increasing_id())
    pred = pred.join(df2, on="id")
    pred.select("date","prediction","close","features")
    
    #run decision tree regression algorithm on data
    dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'close')
    dt_model = dt.fit(train_df)
    dt_pred = dt_model.transform(test_df)
    dt_evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    rmse = dt_evaluator.evaluate(dt_pred)
    stringOutput += (" Decision Tree Regresor RMSE = " + str(rmse))
    
    #run gradient boosted tree regression algorithm on data
    gbt = GBTRegressor(featuresCol = 'features', labelCol = 'close', maxIter=25, maxDepth=3)
    gbt_model = gbt.fit(train_df)
    gbt_pred = gbt_model.transform(test_df)
    gbt_pred.select('prediction', 'close', 'features')
    gbt_evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    rmse = gbt_evaluator.evaluate(gbt_pred)
    stringOutput += (" GBT Regresor RMSE = " + str(rmse))
    print(name +" : "+ stringOutput)

#loop through every file in dataset
for x in file: 
    if(x.isalpha()):
        arg1 = "s3://stockanalysisbucket2.0/full_history/"+x+".csv"
        try:
            df = spark.read.format('csv').load(arg1)
            num = df.count()
            if(num > 7000):
                machineLearningModel(x)
        except:
            print("ops")