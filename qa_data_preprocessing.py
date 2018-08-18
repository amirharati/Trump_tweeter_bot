# qa_data_preprocessing.py
# Amir Harati, Aug 2018
"""
    data preprocessing for question/answers pairs.
"""
import sys

def main(q_file, a_file, bpe_voc_file, word2id, wordid_questions, wordid_answers):

  q_data = [line.strip() for line in open(q_file)]
  a_data = [line.strip() for line in open(a_file)]

  words = [line.split()[0] for line in open(bpe_voc_file)]
  #words = list(set(words))
  words = ["<PAD>", "<START>", "<EOS>"] + words

  words_to_ids = {w: id for id, w in enumerate(words)}
  ids_to_words = {words_to_ids[x]: x for x in words_to_ids}

  with open(word2id, "w") as wif:
    for key, val in words_to_ids.items():
      wif.write(key + "\t" + str(val) + "\n")

  with open(wordid_questions, "w") as f:
    for sen in q_data:
      ostr = ""
      for word in sen.split():
        #print(word)
        ostr = ostr + str(words_to_ids[word]) + " "
      f.write(ostr + "\n")

  with open(wordid_answers, "w") as f:
    for sen in a_data:
      ostr = ""
      for word in sen.split():
        #print(word)
        ostr = ostr + str(words_to_ids[word]) + " "
      f.write(ostr + "\n")


if __name__ == "__main__":
  q_file = sys.argv[1]
  a_file = sys.argv[2]
  bpe_voc_file = sys.argv[3]
  word2id = sys.argv[4]
  wordid_questions = sys.argv[5]
  wordid_answers = sys.argv[6]
  main(q_file, a_file, bpe_voc_file, word2id, wordid_questions, wordid_answers)
