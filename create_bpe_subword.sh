# create_bpe_subword.sh
# Amir Harati Aug 2018
# idea from: https://blog.kovalevskyi.com/how-to-create-a-chatbot-with-tf-seq2seq-for-free-e876ea99063c
# create subwords for chatbot

# uncomment if not cloned.
#git clone https://github.com/rsennrich/subword-nmt.git

subword-nmt/learn_joint_bpe_and_vocab.py --input data/questions.txt data/answers.txt -s 50000 -o data/trump_code.bpe --write-vocabulary data/vocab_questions.bpe data/vocab_answers.bpe
# remove tab  (what about some other chars)
sed -i '/\t/d' data/vocab_questions.bpe 
sed -i '/\t/d' data/vocab_answers.bpe 
cat data/vocab_questions.bpe | cut -f1 --delimiter=' ' > data/revocab_questions.bpe
cat data/vocab_answers.bpe | cut -f1 --delimiter=' ' > data/revocab_answers.bpe

subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_questions.bpe --vocabulary-threshold 10 < data/questions.txt  >  data/questions.bpe
subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_answers.bpe --vocabulary-threshold 10 < data/answers.txt  >  data/answers.bpe


subword-nmt/get_vocab.py  <data/questions.bpe > data/vocab_selected_questions.bpe
subword-nmt/get_vocab.py  <data/answers.bpe > data/vocab_selected_answers.bpe

###  prettrained:  https://github.com/bheinzerling/bpemb