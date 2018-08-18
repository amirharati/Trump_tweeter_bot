# create_english_bpe_dict.sh
# Amir Harati Aug 2018
question=$1
dict=$2
python create_english_bpe_dict.py $question > $dict

