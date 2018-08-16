# qa_data_preprocessing.py
# Amir Harati, Aug 2018
"""
    data preprocessing for question/answers pairs.
"""
outdir = "data/"
q_file = "data/questions.bpe"
a_file = "data/answers.bpe"
bpe_voc_file = "data/vocab_selected_questions_answers.bpe"

q_data = [line.strip() for line in open(q_file)]
a_data = [line.strip() for line in open(a_file)]

words = [line.split()[0] for line in open(bpe_voc_file)]
#words = list(set(words))
words = ["<PAD>", "<START>", "<EOS>"] + words

words_to_ids = {w: id for id, w in enumerate(words)}
ids_to_words = {words_to_ids[x]: x for x in words_to_ids}

with open(outdir + "/qa_word2id.txt", "w") as wif:
  for key, val in words_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/qa_wordid_questions.txt", "w") as f:
  for sen in q_data:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")

with open(outdir + "/qa_wordid_answers.txt", "w") as f:
  for sen in a_data:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")


# short version
q_file = "data/questions.bpe.short"
a_file = "data/answers.bpe.short"
bpe_voc_file = "data/vocab_selected_questions_answers.bpe.short"

q_data = [line.strip() for line in open(q_file)]
a_data = [line.strip() for line in open(a_file)]

words = [line.split()[0] for line in open(bpe_voc_file)]
#words = list(set(words))
words = ["<PAD>", "<START>", "<EOS>"] + words

words_to_ids = {w: id for id, w in enumerate(words)}
ids_to_words = {words_to_ids[x]: x for x in words_to_ids}

with open(outdir + "/qa_word2id.txt.short", "w") as wif:
  for key, val in words_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/qa_wordid_questions.txt.short", "w") as f:
  for sen in q_data:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")

with open(outdir + "/qa_wordid_answers.txt.short", "w") as f:
  for sen in a_data:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")