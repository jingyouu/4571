import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pyspark
%matplotlib inline

from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Read in data
df = pd.read_csv('downloads/ml-20m/ratings.csv',sep = ',',usecols = ['userId','movieId','rating'])
dev = df.sample(n=8000)

# Enable Arrow-based columnar data transfers
spark = SparkSession.builder.appName('pandasToSparkDF').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Create a Spark DataFrame from a pandas DataFrame using Arrow
ratings = spark.createDataFrame(dev)
(training, test) = ratings.randomSplit([0.8,0.2])



# ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

# Grid search
paramGrid = ParamGridBuilder()\
    .addGrid(als.rank, [4,8,12]) \
    .addGrid(als.regParam, [0.1,1,10])\
    .addGrid(als.maxIter, [5,10,15])\
    .addGrid(als.alpha, [1,2,3])\
    .build()
    
# Tune hyper param
tvs = TrainValidationSplit(estimator=als,
                           estimatorParamMaps=paramGrid,
                           evaluator=rmse,
                           trainRatio=0.8)


# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
