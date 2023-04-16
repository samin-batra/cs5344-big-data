## This script is for reading the sentiments file and the
## steam user_item files to calculate the ratings and join them to form 
## a combined user-item matrix which contains a combined rating composed of rating calculated using play_time_forever and sentiments.


import json
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode, percentile_approx, when, col
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import ast
from pyspark.ml.feature import StringIndexer, IndexToString
import pandas as pd
import math
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType



conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "8g").set("spark.driver.memory", "32g").set("spark.sql.execution.arrow.pyspark.enabled", "true").set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

sc = SparkContext(conf=conf)

spark = SparkSession(sc)


#This is for the user items file
user_items = sc.textFile('australian_users_items.json')
user_items = user_items.map(lambda x: ast.literal_eval(x))
user_items = user_items.map(lambda x: (x['user_id'],x['items_count'],x['items']))
user_items_df = user_items.toDF(['user_id','items_count','items'])
user_items_df = user_items_df.select(user_items_df.user_id,user_items_df.items_count,explode(user_items_df.items))
user_items_df = user_items_df.rdd.map(lambda x: (x[0],x[2].get('item_id'),x[2].get('item_name'),x[2].get('playtime_forever'),x[2].get('playtime_2weeks')))
user_items_df = user_items_df.toDF(['user_id','item_id','item_name','playtime_forever','playtime_2weeks'])
games_grouped = user_items_df.groupby("item_id").agg(percentile_approx("playtime_forever",0.5).alias("median"))
user_items_df = user_items_df.join(games_grouped,on='item_id',how='left')
user_items_rdd = user_items_df.rdd.map(lambda x: ((x[0],x[1]),(float(x[3]),float(x[5]))))
user_items_rdd = user_items_rdd.mapValues(lambda x: 5 if x[0]>x[1] else 4 if x[1]>x[0] and x[0]>0.8*x[1] else 3 if 0.8*x[1]>x[0] and x[0]>0.5*x[1] else 2 if 0.5*x[1]>x[0] and x[0]>0.2*x[1] else 1)
user_items_rdd = user_items_rdd.map(lambda x: (x[0][0],x[0][1],x[1]))
user_items_df = user_items_rdd.toDF(['item_id','user_id','rating'])


# #Encoding the user ids to make them suitable for recommendation systems
stringIndexer = StringIndexer(inputCol="user_id",outputCol="user_id_encoded",stringOrderType="frequencyDesc")
model = stringIndexer.fit(user_items_df)
model.setHandleInvalid("error")

stringIndexer.setHandleInvalid("error")
user_items_encoded = model.transform(user_items_df)

user_items_encoded.toPandas().to_csv("user_items_encoded_data.csv")
encoded_ids = user_items_encoded.drop("user_id","item_id","rating")
# user_items_encoded = user_items_encoded.rdd.map(lambda x: (int(x[0]),int(x[2]),int(x[1])))
#inverting to get the mappings for interpreting the 
encoded_ids = encoded_ids.distinct()
# inverter = IndexToString(inputCol="user_id_encoded", outputCol="user_id", labels=model.labels)
# user_id_decoded = inverter.transform(encoded_ids)






##now read the sentiments from sentiment.csv file and join on the original data frame
## containing the implicit ratings
# flag = 0
df_sentiments = spark.read.options(header='True',inferSchema='True', delimiter=',').csv("sentiments.csv")

df_sentiments = df_sentiments.drop("_c0","review","textblob_score","sentiment_class")
combined_ratings_df = user_items_encoded.join(df_sentiments,['user_id','item_id'],how="left")
schema = StructType([StructField("user_id", StringType(), True), StructField("item_id", StringType(), True), StructField("rating",IntegerType()), StructField("user_id_encoded",StringType()), StructField("vader_score",FloatType())])


combined_ratings_df = combined_ratings_df.fillna(-2)

combined_ratings_df = combined_ratings_df.withColumn("vader_score", col("vader_score").cast("float"))
combined_ratings_df = combined_ratings_df.withColumn("rating", col("rating").cast("integer"))


combined_ratings_df = combined_ratings_df.withColumn("vader_score",when(col("vader_score")<0,-1).when(col("vader_score")>0,1).when(col("vader_score")==0,0).otherwise(col("vader_score")))


for col in combined_ratings_df.dtypes:
  print(col[0]+" , " + col[1])


combined_ratings_df = combined_ratings_df.withColumn("ratings",when((combined_ratings_df['rating']>=4) & (combined_ratings_df['vader_score']==1),combined_ratings_df['rating']).
                                                    when((combined_ratings_df["rating"]>=4) & ((combined_ratings_df["vader_score"]==-1) | (combined_ratings_df["vader_score"]==0)),combined_ratings_df["rating"]-1).
                                                    when((combined_ratings_df["rating"]==3) & (combined_ratings_df["vader_score"]==0),combined_ratings_df["rating"]).
                                                    when((combined_ratings_df["rating"]==3) & (combined_ratings_df["vader_score"]==1),combined_ratings_df["rating"]+1).
                                                    when((combined_ratings_df["rating"]==3) & (combined_ratings_df["vader_score"]==-1),combined_ratings_df["rating"]-1).
                                                    when((combined_ratings_df["rating"]>=1) & ((combined_ratings_df["rating"]<=2) & (combined_ratings_df["vader_score"]==-1)),combined_ratings_df["rating"]).
                                                    when((combined_ratings_df["rating"]>=1) & ((combined_ratings_df["rating"]<=2) & (combined_ratings_df["vader_score"]==1)),combined_ratings_df["rating"]+1).
                                                    otherwise(combined_ratings_df["rating"]))
combined_ratings_df = combined_ratings_df.drop("vader_score")
combined_ratings_df.coalesce(1).write.csv("combined_ratings.csv")


## Combining recommendations and implied ratings

user_reviews = sc.textFile('australian_user_reviews.json')
user_reviews = user_reviews.map(lambda x: ast.literal_eval(x))
user_reviews = user_reviews.map(lambda x: (x['user_id'],x['reviews']))
df_user_reviews = user_reviews.toDF(["user_id","reviews"])

# df_user_reviews = spark.read.json("australian_users_reviews.json")
df_user_reviews = df_user_reviews.select(df_user_reviews.user_id,explode(df_user_reviews.reviews))

df_user_reviews = df_user_reviews.rdd.map(lambda x: (x[0],x[1].get('recommend'),x[1].get('item_id')))

df_user_reviews = df_user_reviews.toDF(["user_id","recommended","item_id"])
combined_rec_df = user_items_encoded.join(df_user_reviews,['user_id','item_id'],how="left")
combined_rec_df=  combined_rec_df.withColumn("recommended",when(combined_rec_df["recommended"]=='true',1).when(combined_rec_df["recommended"]=='false',0).otherwise(-1))

combined_rec_df = combined_rec_df.withColumn("recommended", combined_rec_df["recommended"].cast("integer"))
combined_rec_df = combined_rec_df.withColumn("rating", combined_rec_df["rating"].cast("integer"))

combined_rec_df = combined_rec_df.withColumn("ratings",when((combined_rec_df["rating"]<3) & (combined_rec_df['recommended']==1),combined_rec_df['rating']+2).
                                             when((combined_rec_df['rating']==3) & (combined_rec_df['recommended']==1),combined_rec_df['rating']+2).
                                             when((combined_rec_df['rating']==3) & (combined_rec_df['recommended']==0),combined_rec_df['rating']-2).
                                             when((combined_rec_df['rating']>3) & (combined_rec_df['recommended']==0),combined_rec_df['rating']-2 ).
                                             when((combined_rec_df['rating']<3) & (combined_rec_df['recommended']==0),combined_rec_df['rating']).
                                             when((combined_rec_df['rating']>3) & (combined_rec_df['recommended']==1),combined_rec_df['rating']).
                                             otherwise(combined_rec_df['rating']))

combined_rec_df.show(10)
combined_rec_df = combined_rec_df.drop("recommended")

combined_rec_df.coalesce(1).write.csv("recommended_ratings.csv")

