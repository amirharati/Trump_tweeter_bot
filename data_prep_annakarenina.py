"""
  script for data preparation.
"""
import datapreppy


DP2 = datapreppy.DataPreppy("annakarenina_word", "./data/annakarenina_word2id.txt", "./data/annakarenina_wordid_data.txt", "./data")
DP2.save_to_tfrecord()
