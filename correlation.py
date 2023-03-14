import json
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
Step 0.
Setting configurations for py spark to use the default number or cores and 8gig of RAM for execution.
"""
conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "8g").set("spark.driver.memory", "4g")

sc = SparkContext(conf=conf)
spark = SparkSession(sc)


"""
Step 1
Read the data file and the stopwords file. Convert the stopwords file into a list for easier reference.
The data file is a json file which will then be loaded into a dictionary using the json library.
"""
words = sc.textFile(sys.argv[1])
stop_words = sc.textFile(sys.argv[2])
stop_words = stop_words.collect()



"""
Step 2
Here, we load the contents of json file into a dictionary and 
only use the id and abstract fields for now. We remove the special characters, preserve the spaces
and remove the stopwords. Finally, we create an RDD to store the word frequency of words across documents in the form
((id, word), tf)
"""
words = words.map(lambda x: json.loads(x))
words_dict = words.map(lambda x: (x.get('id'),x.get('abstract')))
words_dict = words_dict.flatMapValues(lambda x: (re.split(r'[^\w]+',x)))
words_dict = words_dict.mapValues(lambda x: (x,1))
words_dict = words_dict.map(lambda x: ((x[0],x[1][0].lower()),x[1][1]))
words_dict = words_dict.filter(lambda x: x[0][1] not in stop_words)
counts = words_dict.reduceByKey(lambda n1, n2: n1 + n2)


"""
Step 3
Now, we find the document frequency, i.e. the number of documents that a word occurs in.
We use the same RDD to do so and create a new RDD of the form: (word, DF) where DF is document frequency.
"""
df = counts.map(lambda x:(x[0][1],1))
df = df.reduceByKey(lambda v1,v2: v1+v2)


"""
Step 3
In order to have a vector of values for each document, we need the words which
are not occurring in a given document's abstract. To do that, we perform a catesian product
of the RDD containing the normalized tf-idf values and the RDD containing
all the words across all documents created in Step 3. We filter out the entries where the keys are matching and select
only the distinct rows to get a new RDD of the form:
(id,[words occuring in all abstracts])
"""
new_rdd = counts.cartesian(df) \
    .filter(lambda x: x[0][0] != x[1][0]) \
    .map(lambda x: (x[0][0], x[1][0])).map(lambda x: (x[0][0],x[1]))
new_rdd = new_rdd.distinct().map(lambda x: (x[0],[x[1]]))
new_rdd = new_rdd.reduceByKey(lambda x,y: x+y)



"""
Step 4
We now convert the RDD created in Step 3 into a PySpark dataframe. We then execute an explode
function to allow the array of words mapped to each document ID to spill over into separate rows, each row containing one word.
So, the dataframe takes the form:
--------------------
|id_1      | word1 |
|id_1      | word2 |
|id_1      | word3 |
|.         | .     |
|.         | .     |
|.         | .     |
|.         | .     |
|id_n      |word1  |

We also convert the rdd created in Step 2 into a Py Spark dataframe and then
do a left outer join from the data frame created in this step with the rdd from step 2 that was converted into
a dataframe.
We finally execute some map operations to set none values to 0 for words that occur in the entire corpus
but not in a given document.
We store this in a final RDD in the form: (id,[vector of TF values])
"""
doc_words_df = new_rdd.toDF(['id','words'])

doc_words_df = doc_words_df.select(doc_words_df.id,explode(doc_words_df.words))
doc_words_df = doc_words_df.rdd.map(lambda x:(x[0],x[1])).toDF(['id','words'])

joined_df_df = counts.map(lambda x: (x[0][0],x[0][1],x[1])).toDF(['id','words','tf-idf'])
doc_joined = doc_words_df.join(joined_df_df,on=['id','words'],how='left')

doc_joined = doc_joined.rdd.map(lambda x: (x[0],(x[1],x[2]))).sortBy(lambda x: x[1][0])
doc_joined = doc_joined.mapValues(lambda x: (x[0],0) if x[1] is None else (x[0],x[1]))

final_rdd = doc_joined.map(lambda x: (x[0],[x[1][1]]))
final_rdd = final_rdd.reduceByKey(lambda x,y: x+y)

"""
Step 5
We create a new RDD with the categories and the document ID and
join it to the RDD created in Step 4.
We use only the categories and the vector of TF values from the new RDD in the form:
(category,[vector of tf values for all words occurring in the corpus])
We then perform a reduce operation to group by categories and sum element-wise over the vector of TF values.

"""
categories_dict = words.map(lambda x: (x.get('id'),x.get('categories')))
join_cats = final_rdd.join(categories_dict)
join_cats = join_cats.map(lambda x: (x[1][1],x[1][0]))
join_cats = join_cats.reduceByKey(lambda x,y: [sum(pair) for pair in zip(x,y)])



"""
Step 6
We now do a cartesian product of the RDD created in Step 5 with itself.
We then calculate cosine similarity of all categories with each other using numpy.
"""
final_rdd = join_cats.cartesian(join_cats).map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))

final_rdd = final_rdd.mapValues(lambda x: np.dot(x[0],x[1])/(np.linalg.norm(x[0])*np.linalg.norm(x[1])))
final_rdd = final_rdd.map(lambda x: (x[0][0],(x[0][1],x[1])))


"""
Step 7
From the RDD created in step 6, we create a Pandas dataframe with the columns as the categories and the value
being the correlation value between categories. There are two category columns, both identical since we want to get correlation
of all categories with each other.

We then plot a heatmap using seaborn and save it as a png file.
"""
final_rdd = final_rdd.map(lambda x: (x[0],x[1][0],float(x[1][1])))
final_df = final_rdd.toDF(['category_1','category_2','value'])
pandas_df = final_df.toPandas()
pandas_df = pandas_df.pivot(index='category_1',columns='category_2',values='value')
fig, ax = plt.subplots(figsize=(26,22))         # Sample figsize in inches

sns_plot = sns.heatmap(pandas_df,ax=ax,square=True)
fig = sns_plot.get_figure()
fig.set_size_inches(10,8)
sns_plot.tick_params(labelsize=10)
fig.savefig("heatmap.png",dpi=300, bbox_inches='tight')
