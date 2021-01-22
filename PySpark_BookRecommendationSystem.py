import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf,col,when
import numpy as np


spark = ps.sql.SparkSession.builder.master("local").appName("BookRecommendationSystem").getOrCreate() # read or write.

sc = spark.sparkContext
sqlContext = SQLContext(sc)

ratings_df = spark.read.csv('ratings.csv',header=True,inferSchema=True)

ratings_df.printSchema()

ratings_df.show(5)

books_df = spark.read.csv('books.csv',header=True,inferSchema=True)

books_df.printSchema()

books_df.show(1)

training_df,validation_df = ratings_df.randomSplit([0.8,0.2])

training_df.show(3)
validation_df.show(3)

iterations = 5
regularization_parameter = 0.1
rank = 4
errors = []
err = 0


als = ALS(maxIter=iterations,regParam=regularization_parameter,rank=4,userCol="user_id",ratingCol="rating",itemCol="book_id")
model = als.fit(training_df)
predictions = model.transform(validation_df)
new_predictions = predictions.filter(col('prediction')!=np.nan)
evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
rmse = evaluator.evaluate(new_predictions)
print("RMSE",rmse)



iterations = 10
regularization_parameter = 0.1
rank = 5
errors = []
err = 0


als = ALS(maxIter=iterations,regParam=regularization_parameter,rank=4,userCol="user_id",ratingCol="rating",itemCol="book_id")
model = als.fit(training_df)
predictions = model.transform(validation_df)
new_predictions = predictions.filter(col('prediction')!=np.nan)
evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
rmse = evaluator.evaluate(new_predictions)
print("RMSE",rmse)


# Using Cross Validation, takes a longer time to run

ls = ALS(maxIter=iterations,regParam=regularization_parameter,rank=rank,userCol="user_id",ratingCol="rating",itemCol="book_id")

paramGrid = ParamGridBuilder()\
.addGrid(als.regParam,[0.1,0.01,0.18])\
.addGrid(ls.rank,range(4,10)).build()

evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
crossval = CrossValidator(estimator = ls,
                         estimatorParamMaps=paramGrid,
                         evaluator=evaluator,
                         numFolds=5)
cvModel = crossval.fit(training_df)

#Predictions

predictions = model.transform(validation_df)
predictions.show(10)
