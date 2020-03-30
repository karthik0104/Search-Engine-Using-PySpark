#!/usr/bin/env python
# coding: utf-8

import os
import sys

if __name__ == "__main__":

    
    # Initialize Spark context and Conf

    from pyspark import SparkContext, SparkConf

    conf = SparkConf().setAppName("search")
    sc = SparkContext(conf=conf)

    ############################## PART I Starts Here ##################################

    # Read shakespeare_small.json

    import json
    import requests

    from pyspark.sql import SQLContext

    sqlContext = SQLContext(sc)

    r = requests.get("http://elmokhtari.com/downloads/ds8003/shakespeare_small.json")
    df1 = sqlContext.createDataFrame([json.loads(line) for line in r.iter_lines()])

    df1.show()

    # Show row count
    print('Rows count = ' + str(df1.count()))

    # Load shakespeare_full.json and show top 20 rows
    rddjson = sc.textFile('shakespeare_full.json')
    df2 = sqlContext.read.json(rddjson)

    df2.show()

    # Q.4. Show count of entries grouped by "speaker"
    df2_speaker_groups = df2.groupBy("speaker").count()
    df2_speaker_groups.show(df2_speaker_groups.count(), False)

    # Register df2 as temp table
    df2.registerTempTable("speeches")

    # Using spark.sql, show all entries where line_number starts with “1.1.” and text_entry contains the word “sometimes”.

    df_filtered = sqlContext.sql('''
    SELECT * FROM speeches
    WHERE line_number like '1.1.%'
    AND
    text_entry like '%sometimes%'
    ''')
    df_filtered.show()

    # Generate a list with the number of characters in every text entry where the speaker is “DONALBAIN”

    df_filtered2 = sqlContext.sql('''
    SELECT * FROM speeches
    WHERE speaker = 'DONALBAIN'
    ''')

    from pyspark.sql.functions import *

    df_filtered3 = df_filtered2.withColumn("num_text_entry_chars", length(col('text_entry')))
    num_chars_list = df_filtered3.select('num_text_entry_chars').rdd.flatMap(lambda x: x).collect()

    print(num_chars_list)

    # Consider all text entries of the speaker “DONALBAIN”. Generate a list of pairs (key, value) where key is the _id of the text entry and value is the number of words in this text entry.

    df_filtered4 = df_filtered2.withColumn('wordCount', size(split(col('text_entry'), ' '))).select('_id', 'wordCount')
    df_filtered4 = df_filtered4.rdd.collectAsMap()

    print(df_filtered4)

    ############################## PART II Starts Here ##################################

    # Raw data
    df2.show()

    # Perform pre-processing steps

    # Step 1: Convert words to lower-case
    df2_processed = df2.withColumn('text_entry', lower(col('text_entry')))

    # Step 2: Remove punctuations
    df2_processed = df2_processed.withColumn('text_entry', trim(regexp_replace('text_entry', '[^A-Za-z0-9 ]+', '')))

    # Processed Data
    df2_processed.show()

    # Split sentence into bag of words and store it into column (Basically, create document corpus)
    from pyspark.sql.functions import split

    document_corpus = df2_processed.withColumn("words", split("text_entry", "\s+")).select('_id', 'words')

    # Unfold document corpus. Basically, it means to create document-word combo from the bag of words.
    unfoldedDocs = document_corpus.select(col('_id').alias('doc_id'), 'words', explode('words').alias('token'))
    unfoldedDocs.show()

    # Q.1. Compute TFIDF scores for all words in all text entries and build an inverted index.

    # Compute TF
    intermediate_df1 = unfoldedDocs.groupBy("doc_id", "token").agg(count("words").alias('tf'))

    # Compute DF
    intermediate_df2 = unfoldedDocs.groupBy("token").agg(countDistinct("doc_id").alias('df'))

    # Get document count
    doc_count = unfoldedDocs.select('doc_id').distinct().count()

    # Computer IDF
    intermediate_df3 = intermediate_df2.withColumn('idf', log((doc_count + 1) / (col('df') + 1)))

    # Join the intermediate tables
    tokensWithTfIdf = intermediate_df1.join(intermediate_df3, intermediate_df1.token == intermediate_df3.token,
                                            how="left").select(intermediate_df1["*"], *intermediate_df3.drop("token"))

    # Compute tf-idf and Cache it
    tokensWithTfIdf = tokensWithTfIdf.withColumn('tf_idf', col('tf') * col('idf')).cache()  # Caching performed
    tokensWithTfIdf.show()


    # Construct function search_words(query, N)
    def search_words(query, N):
        # Split query into list of query words
        q_words = query.split()

        # Filter tokens based on query word list
        q_tokens = tokensWithTfIdf.filter(col('token').isin(q_words))

        # Aggregate based on documents and calculate sum of tf-idf to calculate document score
        agg_tf_idf = q_tokens.groupby('doc_id').sum('tf_idf').select('doc_id', col('sum(tf_idf)').alias('sum_tf_idf'))

        # For each document, find out the proportion of query words present in it
        cnt_tf_idf = q_tokens.groupby('doc_id').count().select(col('doc_id').alias('doc_id2'),
                                                               (col('count') / len(q_words)).alias('count_tf_idf'))

        doc_score = agg_tf_idf.join(cnt_tf_idf, agg_tf_idf.doc_id == cnt_tf_idf.doc_id2, how="left").select(agg_tf_idf["*"],
                                                                                                            cnt_tf_idf[
                                                                                                                "count_tf_idf"])

        doc_score = doc_score.withColumn('doc_score', col('sum_tf_idf') * col('count_tf_idf'))
        doc_score = doc_score.sort(col('doc_score').desc()).limit(N)

        results = doc_score.join(df2, doc_score.doc_id == df2._id, how="left").select(doc_score["doc_id"],
                                                                                      doc_score["doc_score"],
                                                                                      df2["text_entry"])
        results.sort(col('doc_score').desc()).show(results.count(), False)


    search_words("if you said so", 5)
