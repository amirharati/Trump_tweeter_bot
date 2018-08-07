# create_bpe_subword.sh
# Amir Harati Aug 2018
# idea from: https://blog.kovalevskyi.com/how-to-create-a-chatbot-with-tf-seq2seq-for-free-e876ea99063c
# create subwords for chatbot

# uncomment if not cloned.
#git clone https://github.com/rsennrich/subword-nmt.git

subword-nmt/learn_joint_bpe_and_vocab.py --input data/text_data.txt  -s 50000 -o tmp_code.bpe --write-vocabulary temp.a
# remove tab  (what about some other chars)
sed -i '/\t/d' ./temp.a
cat temp.a | cut -f1 --delimiter=' ' > revocab.temp.a
subword-nmt/apply_bpe.py -c tmp_code.bpe --vocabulary temp.a --vocabulary-threshold 10 < data/text_data.txt >  text_temp

subword-nmt/get_vocab.py  <text_temp > temp2.a