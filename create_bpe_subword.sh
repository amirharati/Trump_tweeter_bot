# create_bpe_subword.sh
# Amir Harati Aug 2018
# idea from: https://blog.kovalevskyi.com/how-to-create-a-chatbot-with-tf-seq2seq-for-free-e876ea99063c
# create subwords for chatbot

# uncomment if not cloned.
#git clone https://github.com/rsennrich/subword-nmt.git

# put them together so we can share the vocab. btween encoder/decoder.
cat data/questions.txt data/answers.txt > data/questions_answers.txt

subword-nmt/learn_joint_bpe_and_vocab.py --input data/questions_answers.txt -s 50000 -o data/trump_code.bpe --write-vocabulary data/vocab_questions_answers.bpe
# remove tab  (what about some other chars)
sed -i '/\t/d' data/vocab_questions_answers.bpe 
#sed -i '/\t/d' data/vocab_answers.bpe 
#cat data/vocab_questions.bpe | cut -f1 --delimiter=' ' > data/revocab_questions.bpe
#cat data/vocab_answers.bpe | cut -f1 --delimiter=' ' > data/revocab_answers.bpe

# we select at least 500 since we inflate the data 
subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_questions_answers.bpe --vocabulary-threshold 500 < data/questions.txt  >  data/questions.bpe
subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_questions_answers.bpe --vocabulary-threshold 500 < data/answers.txt  >  data/answers.bpe

cat data/questions.bpe data/answers.bpe > data/questions_answers.bpe

subword-nmt/get_vocab.py  <data/questions_answers.bpe > data/vocab_selected_questions_answers.bpe


#subword-nmt/get_vocab.py  <data/answers.bpe > data/vocab_selected_answers.bpe


# TODO: use pretrained model.
###  prettrained:  https://github.com/bheinzerling/bpemb


# short version
cat data/questions.txt.short data/answers.txt.short > data/questions_answers.txt.short

subword-nmt/learn_joint_bpe_and_vocab.py --input data/questions_answers.txt.short -s 50000 -o data/trump_code.bpe.short --write-vocabulary data/vocab_questions_answers.bpe.short
# remove tab  (what about some other chars)
sed -i '/\t/d' data/vocab_questions_answers.bpe.short


# we select at least 500 since we inflate the data 
subword-nmt/apply_bpe.py -c data/trump_code.bpe.short --vocabulary data/vocab_questions_answers.bpe.short --vocabulary-threshold 10 < data/questions.txt.short  >  data/questions.bpe.short
subword-nmt/apply_bpe.py -c data/trump_code.bpe.short --vocabulary data/vocab_questions_answers.bpe.short --vocabulary-threshold 10 < data/answers.txt.short  >  data/answers.bpe.short

cat data/questions.bpe.short data/answers.bpe.short > data/questions_answers.bpe.short

subword-nmt/get_vocab.py  <data/questions_answers.bpe.short > data/vocab_selected_questions_answers.bpe.short