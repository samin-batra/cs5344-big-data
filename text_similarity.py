import json
import re
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, explode
import math
import numpy as np



"""
Step 0.
Setting configurations for py spark to use the default number or cores and 8gig of RAM for execution.
"""
conf = SparkConf().setMaster("local[*]").set("spark.executor.memory", "8g").set("spark.driver.memory", "4g")

sc = SparkContext(conf=conf)

"""
Step 1
Read the data file and the stopwords file. Convert the stopwords file into a list for easier reference.
The data file is a json file which will then be loaded into a dictionary using the json library.
"""
words_file = sys.argv[1]
stopwords_file = sys.argv[2]
words = sc.textFile(words_file)
stop_words = sc.textFile(stopwords_file)
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

words_dict = words_dict.mapValues(lambda x: re.sub('\W+',' ', x))
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
Step 4
In order to calculate the TF-IDF values for words occurring in each document abstract, 
we merge the two RDDs calculated in the previous two steps into a single RDD.
That new RDD is of the form: ((id, word),(tf, df)) where ID-> document ID, tf-> term frequency of word in document and df-> is the document frequency of the word

"""
counts = counts.map(lambda x: (x[0][0],x[0][1],x[1]))
spark = SparkSession(sc)

counts = counts.map(lambda x: (x[1],(x[0],x[2])))

joined_df = counts.join(df).map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))


"""
Step 5
We also get the number of documents and store in a variable. This 
will be used to calculate TF-IDF.
"""
num_docs = words.map(lambda x: x.get('id')).count()

"""
Step 6
Here, we calculate the TF-IDF and the normalized TF-IDF values. We first calculate the TF-IDF values
by running a mapvalues function on the RDD formed in Step 4. We then create a new RDD to sum over the squared TF-IDF values in each document 
and then take their square root.
We the perform a left outer join on the RDD containing the TF-IDF and the RDD containing the square root
of the sum of squares of TF-IDF values over each document which is now referred to as t.
The new RDD is of the form: ((id,word),(tf-idf,t)).
We then execute a map function to calculate normalized tf-idf as: tf-idf/t
"""

joined_df = joined_df.mapValues(lambda x: ((1+math.log(x[0],10))*math.log(num_docs/x[1],10)))
normalized_tf_idf = joined_df.map(lambda x: (x[0][0],x[1]))
normalized_tf_idf = normalized_tf_idf.mapValues(lambda x: math.pow(x,2))
normalized_tf_idf = normalized_tf_idf.reduceByKey(lambda x,y: x+y)
normalized_tf_idf = normalized_tf_idf.mapValues(lambda x: math.sqrt(x))
joined_df = joined_df.map(lambda x: (x[0][0],(x[0][1],x[1])))
joined_df = joined_df.leftOuterJoin(normalized_tf_idf)
joined_df = joined_df.map(lambda x: ((x[0],x[1][0][0]),(x[1][0][1]/x[1][1])))


"""
Step 7
In order to have a vector of values for each document, we need the words which
are not occurring in a given document's abstract. To do that, we perform a catesian product
of the RDD containing the normalized tf-idf values and the RDD containing
all the words across all documents created in Step 3. We filter out the entries where the keys are matching and select
only the distinct rows to get a new RDD of the form:
(id,[words occuring in all abstracts])
"""
new_rdd = joined_df.cartesian(df) \
    .filter(lambda x: x[0][0] != x[1][0]) \
    .map(lambda x: (x[0][0], x[1][0])).map(lambda x: (x[0][0],x[1]))
new_rdd = new_rdd.distinct().map(lambda x: (x[0],[x[1]]))
new_rdd = new_rdd.reduceByKey(lambda x,y: x+y)



"""
Step 8
We now convert the RDD created in Step 7 into a PySpark dataframe. We then execute an explode
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

Now, we convert it back to an RDD of the form ((id,word),0). The zero is put to make it uniform with the rdd created
in Step 6.
Finally, we perform a left outer join from the new RDD to the RDD created in Step 6. We then map the 'None'
values generated to 0 as this means those words don't occur in the abstract of the given document.
Finally, we convert this RDD back to the form:
(id,[vector of words with normalized tf-idf values])
This RDD will be used in the final step of this program to calculate cosine similarity.
"""
doc_words_df = new_rdd.toDF(['id','words'])

doc_words_df = doc_words_df.select(doc_words_df.id,explode(doc_words_df.words))
doc_words_df = doc_words_df.rdd.map(lambda x:((x[0],x[1]),0))

joined_df_df = joined_df.map(lambda x: ((x[0][0],x[0][1]),x[1]))
doc_joined = doc_words_df.leftOuterJoin(joined_df_df)
doc_joined = doc_joined.map(lambda x: (x[0][0],(x[0][1],x[1][1]))).sortBy(lambda x: x[1][0])
doc_joined = doc_joined.mapValues(lambda x: (x[0],0) if x[1] is None else (x[0],x[1]))

final_rdd = doc_joined.map(lambda x: (x[0],[x[1][1]]))
final_rdd = final_rdd.reduceByKey(lambda x,y: x+y)

final_rdd = final_rdd.sortBy(lambda x: x[1][0])



"""
Step 9
We perform the following steps in the same way as we performed the steps for abstracts.
The only difference is that TF here is the term frequency of the word occurring in the title of a document.
This step is similar to Step 2.
"""
title_dict = words.map(lambda x: (x.get('id'),x.get('title')))

title_dict = title_dict.flatMapValues(lambda x: (re.split(r'[^\w]+',x)))
title_dict = title_dict.mapValues(lambda x: (x,1))
title_dict = title_dict.map(lambda x: ((x[0],x[1][0].lower()),x[1][1]))

title_dict = title_dict.filter(lambda x: x[0][1] not in stop_words)
title_tf = title_dict.reduceByKey(lambda n1, n2: n1 + n2)

title_tf = title_tf.map(lambda x: (x[0][1],(x[0][0],x[1])))



"""
Step 10
In this step we calculate the TF-IDF values for each word occurring in the title of a document.
This is similar to Steps 4 and 6. In order to account for words which occur in the title 
but not in the abstract, we set their TF-IDF values to 0.
We then calculate the normalized TF-IDF values.
"""
joined_title_df = title_tf.leftOuterJoin(df)

joined_title_df = joined_title_df.map(lambda x: ((x[1][0][0],x[0]),(x[1][0][1],x[1][1])))

joined_title_df = joined_title_df.map(lambda x: ((x[0][0],x[0][1],x[1][0]),x[1][1])).mapValues(lambda x: 0 if x is None else x)
joined_title_df = joined_title_df.map(lambda x: ((x[0][0],x[0][1]),(x[0][2],x[1])))


joined_title_df = joined_title_df.mapValues(lambda x: (0 if x[1]==0  else (1+math.log(x[0],10))*math.log(num_docs/x[1],10)))
normalized_title_tf_idf = joined_title_df.map(lambda x: (x[0][0],x[1]))
normalized_title_tf_idf = normalized_title_tf_idf.mapValues(lambda x: math.pow(x,2))
normalized_title_tf_idf = normalized_title_tf_idf.reduceByKey(lambda x,y: x+y)
normalized_title_tf_idf = normalized_title_tf_idf.mapValues(lambda x: math.sqrt(x))

joined_title_df = joined_title_df.map(lambda x: (x[0][0],(x[0][1],x[1])))
joined_title_df = joined_title_df.join(normalized_title_tf_idf)
joined_title_df = joined_title_df.map(lambda x: ((x[0],x[1][0][0]),(0)) if x[1][1]==0 else((x[0],x[1][0][0]),(x[1][0][1]/x[1][1])))


"""
Step 11
This step is the same as Step 7. We need a separate RDD which contains all the words occurring
across all abstracts to be mapped to each document ID.
We discard those words which are occurring in title but not in abstract to maintain the equality of vectors for
calculating cosine similarity.
"""
new_title_rdd = joined_title_df.cartesian(df) \
    .filter(lambda x: x[0][0] != x[1][0]) \
    .map(lambda x: (x[0][0], x[1][0])).map(lambda x: (x[0][0],x[1]))
new_title_rdd = new_title_rdd.distinct().map(lambda x: (x[0],[x[1]]))
new_title_rdd = new_title_rdd.reduceByKey(lambda x,y: x+y)

title_words_df = new_title_rdd.toDF(['id','words'])
title_words_df = title_words_df.select(title_words_df.id,explode(title_words_df.words))
title_words_df = title_words_df.rdd.map(lambda x:((x[0],x[1]),0))

joined_title_df_df = joined_title_df.map(lambda x: ((x[0][0],x[0][1]),x[1]))

title_joined = title_words_df.leftOuterJoin(joined_title_df_df)

title_joined = title_joined.map(lambda x: (x[0][0],(x[0][1],x[1][1]))).sortBy(lambda x: x[1][0])
title_joined = title_joined.mapValues(lambda x: (x[0],0) if x[1] is None else (x[0],x[1]))
final_title_rdd = title_joined.map(lambda x: (x[0],[x[1][1]]))
final_title_rdd = final_title_rdd.reduceByKey(lambda x,y: x+y)



"""
Step 12
This is the final step of the program. We calculate the cosine similarity for each title and each abstract across documents
by first taking a cartesian product, then using numpy to calculate the
dot product and the multiplication of the L2-norms of the vectors of title and abstract across documents.
We then pick the top-1 value which has the highest cosine similarity for each title. The rest of them are discarded since 
we are only trying to get the top-1 search results.
Finally, we encode these as 1 or 0 (1 meaning that the title matched to its abstract, 0 meaning that it did not match.)
In this way, we calculate the accuracy of the algorithm by summing over all 1s and then dividing by the total number of documents.
We get an accuracy of 91%.
"""
final_rdd = final_rdd.cartesian(final_title_rdd).map(lambda x: ((x[0][0],x[1][0]),(x[0][1],x[1][1])))

final_rdd = final_rdd.mapValues(lambda x: np.dot(x[0],x[1])/(np.linalg.norm(x[0])*np.linalg.norm(x[1])))
final_rdd = final_rdd.map(lambda x: (x[0][0],(x[0][1],x[1])))

final_rdd = final_rdd.reduceByKey(lambda x,y: x if x[1]>y[1] else y)
acc_score = final_rdd.map(lambda x: 1 if x[0]==x[1][0] else 0)

final_score = acc_score.sum()/num_docs
final_score = final_score*100
sc.parallelize([('Accuracy score is',final_score)]).repartition(1).saveAsTextFile("final-accuracy.txt")
