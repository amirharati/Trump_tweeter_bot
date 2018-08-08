# find_freq.sh
# Amir Harati Aug 2018
# find freq n-gram to be used as input for chatbot.
# we generate a list and then filter out based on it (the program generate the input will map some of these to <UNK>)

~/kaldi/tools/srilm/bin/i686-m64/ngram-count   -text data/text_data.txt -order 4 -write corpus.count

awk '{print $NF,$0}' corpus.count | sort -nr | cut -f2- -d' ' |head -n 30000 > freq_grams.list