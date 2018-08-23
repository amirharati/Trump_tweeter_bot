# apply_bpe.sh
# apply the bpe model to new data

questions=$1
answers=$2
questions_bpe=$3
answers_bpe=$4
T=$5

# we select at least 500 since we inflate the data 
subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_questions_answers.bpe --vocabulary-threshold  $T < $questions  >  $questions_bpe
subword-nmt/apply_bpe.py -c data/trump_code.bpe --vocabulary data/vocab_questions_answers.bpe --vocabulary-threshold $T < $answers  >  $answers_bpe
