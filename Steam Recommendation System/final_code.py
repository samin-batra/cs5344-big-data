import json
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, percentile_approx
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import ast
from pyspark.ml.feature import StringIndexer, IndexToString
import pandas as pd
import math


conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "8g").set("spark.driver.memory", "32g").set("spark.sql.execution.arrow.pyspark.enabled", "true").set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

sc = SparkContext(conf=conf)

spark = SparkSession(sc)

user_items_encoded = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("user_items_encoded_data.csv")
user_items_encoded.printSchema()


DATA_USE = 3 #1 for without sentiments, 2 for with sentiments, 3 for using recommendations

if DATA_USE==1:
    ## Data without sentiments
    ratings_raw_data = sc.textFile("user_items_encoded_data.csv")
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    user_items_encoded = ratings_raw_data.filter(lambda line: line != ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (int(float(tokens[4])),int(tokens[1]),int(float(tokens[3])))).cache()

elif DATA_USE==2:
    ## Data with sentiments 
    ratings_raw_data = sc.textFile("ratings.csv")
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    user_items_encoded = ratings_raw_data.map(lambda line: line.split(",")).map(lambda tokens: (int(float(tokens[3])),int(tokens[1]),int(float(tokens[4])))).cache()

elif DATA_USE==3:
    ## data with recommendations and implicit ratings
    ratings_raw_data = sc.textFile("recommended_1.csv")
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    user_items_encoded = ratings_raw_data.map(lambda line: line.split(",")).map(lambda tokens: (int(float(tokens[3])),int(tokens[1]),int(float(tokens[4])))).cache()


rddTraining, rddTesting = user_items_encoded.randomSplit([8,2], seed=1001)

ranks = [10,20,30,40,50,60]   


mse = []

for rank in ranks:

    numIterations = 10

    #Iteration numbers



    alpha=0.01


    #Confidence values in ALS，default 1.0

    #lambda
    #Regularization parameter，DEFAULT 0.01

    ############################################################################################
    # Build the recommendation model using Alternating Least Squares based on implicit ratings #
    ############################################################################################

    model = ALS.trainImplicit(rddTraining, rank, 10, alpha=0.01)


    # testset = sc.parallelize([(3, 4000), (3, 15700)])   
    ##########   you can assume any user_id and any item(game) id here   ######################
    # model.predictAll(testset).collect()

    # Calculate all predictions
    rddTesting_map = rddTesting.map(lambda r: ((r[0], r[1]))) 
    predictions = model.predictAll(rddTesting_map).map(lambda r: ((r[0], r[1]), (r[2]))) 
    predictions.take(5)    ####### Output 5 results
    # predictions.repartition(1).saveAsTextFile("predictions.txt")


    rates_and_preds = rddTesting.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions) 
    # rates_and_preds.repartition(1).saveAsTextFile("ratings_preds.txt")


    Spark_rec_list = []
    for i in range(8):
        Spark_rec_list.append(model.recommendProducts(i,10))
    Spark_rec_df = pd.DataFrame(Spark_rec_list)

    Spark_rec_df.to_csv("top_recommendations_" + str(rank) +".csv")

    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()) 
    print ('For testing data the RMSE is %s' % (error))
    mse.append(error)

mse_comparison = pd.DataFrame({'num latent features':ranks,'mse':mse})
mse_comparison.to_csv("mse_comparison.csv")

