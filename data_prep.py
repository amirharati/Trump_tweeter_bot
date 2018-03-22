"""
  script for data preparation.
"""
import datapreppy

DP1 = datapreppy.DataPreppy("char", "./data/chars2id.txt", "./data/charid_data.txt", "./data")
DP1.save_to_tfrecord()

DP2 = datapreppy.DataPreppy("word", "./data/word2id.txt", "./data/wordid_data.txt", "./data")
DP2.save_to_tfrecord()
