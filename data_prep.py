"""
extract and normalize tweets from raw data and save the resutls.

Amir Harati, March 2018
"""
import json
import textacy as tc
import glob, os

json_data = dict()
os.chdir("./raw_data")
for file in glob.glob("master*.json"):
  with open(file) as data_file:
    json_data[file] = json.load(data_file)

# read and normalize the tweets
data = list()
for k in json_data:
  for d in json_data[k]:
    try:
      data.append(tc.preprocess.preprocess_text(d["full_text"],
       transliterate=True, no_urls=True, no_emails=True))
    except:
      data.append(tc.preprocess.preprocess_text(d["text"],
       transliterate=True, no_urls=True, no_emails=True))

# next tokenize for word based LM (and do other needed things)
# anohter version for char based
# use Spacy word2vec (or something else) to convert words to vec
# both for input to RNN and for conditional
