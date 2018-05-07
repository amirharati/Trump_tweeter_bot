"""
extract and normalize tweets from raw data and save the resutls.

Amir Harati, March 2018
issues:
  1- chunking is not used (e.g. new york would be new + york)
"""
import json
import textacy as tc
import glob, os
import spacy as sp
import unicodedata

outdir = "./data"
#nlp = sp.load('en_core_web_lg')

#nlp = sp.load('en')

json_data = dict()
os.chdir("./raw_data")
for file in glob.glob("master*.json"):
  with open(file) as data_file:
    json_data[file] = json.load(data_file)

os.chdir("..")

# read and normalize the tweets
data = list()
for k in json_data:
  for d in json_data[k]:
    try:
      temp = d["full_text"]
    except:
      temp = d["text"]

    temp = str(unicodedata.normalize('NFKD', temp).encode('ascii', 'ignore'))
    temp = temp.strip("\r\n")
    temp = temp.replace("\n", " ")
    temp = temp.replace("\t", " ")
    temp = tc.preprocess.normalize_whitespace(tc.preprocess.preprocess_text(temp, transliterate=True, no_urls=True, no_emails=True))
    temp = temp.replace("*EMAIL*", "")
    temp = temp.replace("*URL*", "")
    temp = temp.replace("&amp", "")
    temp = temp.replace("&lt", "")
    temp = temp.replace("\"", "")
    temp = temp.replace("\'", "")
    temp = temp.replace("&gt", "")
    temp = temp.replace("-", "")
    #temp = temp.replace(".", "")
    temp = temp.replace("    ", " ")
    temp = temp.replace("   ", " ")
    temp = temp.replace("  ", "")
    temp = temp.replace('?', ' ? ')
    temp = temp.replace('!', ' ! ')

    data.append(temp)


words = set()
chars = set()
# add <START> and <EOS> to the vocs.
#words.add("<START>")
#words.add("<EOS>")
#chars.add("<START>")
#chars.add("<EOS>")
#  For now it  is  not acally used.

#count = 0
#tweet_to_tokens = {}
#for tweet in data:
#  print(count)
#  count += 1
#  doc = nlp(tweet)
#  tweet_to_tokens[tweet] = doc
#  le = [x for x in tweet]
#  for c in le:
#    chars.add(c)
#  for token in doc:
#    words.add(token.text)

#data = [item for sublist in data for item in sublist]

#print(data[0:1000])

speeches = [line.strip() for line in open("data/trump_speeches.txt")]
sdata = []
for line in speeches:
  temp = line.strip("\r\n")
  temp = temp.replace("\n", " ")
  temp = temp.replace("\t", " ")
  temp = tc.preprocess.normalize_whitespace(tc.preprocess.preprocess_text(temp, transliterate=True, no_urls=True, no_emails=True))
  temp = temp.replace("*EMAIL*", "")
  temp = temp.replace("*URL*", "")
  temp = temp.replace("&amp", "")
  temp = temp.replace("&lt", "")
  temp = temp.replace("\"", "")
  temp = temp.replace("\'", "")
  temp = temp.replace("&gt", "")
  temp = temp.replace("-", "")
  #temp = temp.replace(".", "")
  temp = temp.replace("    ", " ")
  temp = temp.replace("   ", " ")
  temp = temp.replace("  ", "")
  temp = temp.replace('?', ' ? ')
  temp = temp.replace('!', ' ! ')

  sdata.append(temp)

# add speeches to  tweets
data = data + sdata

line = ""
pdata = []
words = set()
chars = set()
for l in data:
  for w in l.split():
    words.add(w)
    for c in w:
      chars.add(c)
    #line += w + " "
  #else:
  pdata.append(l)
  #line = ""


chars = list(chars)
words = list(words)

words = sorted(words)
chars = sorted(chars)
words = ["<PAD>", "<START>", "<EOS>"] + words
chars = ["<PAD>", "<START>", "<EOS>", " "] + chars

print("#chars: ", len(chars))
print("#words:", len(words))

words_to_ids = {w: id for id, w in enumerate(words)}
ids_to_words = {words_to_ids[x]: x for x in words_to_ids}
chars_to_ids = {w: id for id, w in enumerate(chars)}
ids_to_chars = {chars_to_ids[x]: x for x in chars_to_ids}


# save data
with open(outdir + "/word2id.txt", "w") as wif:
  for key, val in words_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/chars2id.txt", "w") as wif:
  for key, val in chars_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/text_data.txt", "w") as f:
  for tweet in pdata:
    f.write(tweet + "\n")

with open(outdir + "/charid_data.txt", "w") as f:
  for tweet in pdata:
    ostr = ""
    #le = [x for x in tweet]
    for c in tweet:
      ostr = ostr + str(chars_to_ids[c]) + " "
    f.write(ostr + "\n")


with open(outdir + "/wordid_data.txt", "w") as f:
  for sen in pdata:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")

# with open(outdir + "/wordid_data.txt", "w") as f:
#   for tweet in pdata:
#     ostr = ""
#     for token in tweet_to_tokens[tweet]:
#       #print(word)
#       word = token.text
#       ostr = ostr + str(words_to_ids[word]) + " "
#     f.write(ostr + "\n")


# another script for dividing to train/test sets


# outputs: 1 raw tweets, 2 tweets in integer format
# 3 list of non-stop words (also entety recogion?) in both raw and integer for each tweet
# Do we need further normalization like remove @xxx ?
#

# next tokenize for word based LM (and do other needed things)
# anohter version for char based
# use Spacy word2vec (or something else) to convert words to vec
# both for input to RNN and for conditional
