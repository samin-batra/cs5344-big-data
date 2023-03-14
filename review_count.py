import sys
from pyspark import SparkConf, SparkContext
import ast
import json

conf = SparkConf()

sc = SparkContext(conf=conf)

#step 1
#reading the reviews file and loading as json
#running map to get the 4 fields (asin, review time, review count, overall)
#running reduce by key to calculate total number of reviews for a pair (asin, reviewtime) and the average rating
video_games = sc.textFile(sys.argv[1])
video_games_dict = video_games.map(lambda x: json.loads(x))


mapped_vg = video_games_dict.map(lambda x: (((x.get('asin'), x.get('reviewTime')), (1,x.get('overall')))))

agg_vg = mapped_vg.reduceByKey(lambda v,w: (v[0]+w[0],v[1]+w[1])).mapValues(lambda v: (v[0],round(v[1]/v[0],2)))

#step 2
#read the meta file and get the asin and bran names
meta_vg = sc.textFile(sys.argv[2])
meta_vg_dict = meta_vg.map(lambda x: ast.literal_eval(x))

meta_vg_map = meta_vg_dict.map(lambda x: (x.get('asin'),x.get('brand') if 'brand' in x else None))
meta_vg_map = meta_vg_map.filter(lambda x: x[1] != None)

#step 3
# joining the reviews and the meta rdd
new_agg_vg = agg_vg.map(lambda x: (x[0][0],(x[0][1],x[1])))
joined_map = new_agg_vg.join(meta_vg_map)

#step 4
#sorting the records to find the products with highest number of reviews in a day
new_join_map = joined_map.map(lambda x: ((x[0],x[1][0][0],x[1][0][1][0],x[1][0][1][1],x[1][1])))
vals_rdd = sc.parallelize(new_join_map.sortBy(ascending=False,keyfunc=lambda x: x[2]).take(15))

#step 5
#output the reviews to a file
vals_rdd.repartition(1).saveAsTextFile(sys.argv[3])

sc.stop()
