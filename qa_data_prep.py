import Seq2SeqDataPreppy as sdp


DP4 = sdp.Seq2SeqDataPreppy("qa_word", "./data/qa_word2id.txt", "./data/qa_wordid_questions.txt", "./data/qa_wordid_answers.txt", "./data")
DP4.save_to_tfrecord()

# short version
DP5 = sdp.Seq2SeqDataPreppy("qa_word_short", "./data/qa_word2id.txt.short", "./data/qa_wordid_questions.txt.short", "./data/qa_wordid_answers.txt.short", "./data")
DP5.save_to_tfrecord()

#DP5 = datapreppy.DataPreppy("qa_word_answers", "./data/qa_word2id.txt", "./data/qa_wordid_answers.txt", "./data")
#DP5.save_to_tfrecord()