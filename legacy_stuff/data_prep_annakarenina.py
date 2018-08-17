"""
  script for data preparation.
"""
import datapreppy

DP1 = datapreppy.DataPreppy("annakarenina_echar", "./data/annakarenina_echars2id.txt", "./data/annakarenina_echarid_data.txt", "./data")
DP1.save_to_tfrecord()

DP2 = datapreppy.DataPreppy("annakarenina_word", "./data/annakarenina_word2id.txt", "./data/annakarenina_wordid_data.txt", "./data")
DP2.save_to_tfrecord()
