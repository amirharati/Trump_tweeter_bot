# create_dialouge_dataset_pipeline.sh
# Amir Harati, Aug 2018
# Create data needed to build chatbot. 
# The input to this include set of question/answer pairs and output is dataset in tfrecord format and related files (dict etc)

# update the following vars to generate new datasets
# e.g. for using a commbined dataset for initilization we need to create question/answers pairs and replace here.



# trump + corrnelmovies (for initilization), We also use vocabs from this dataset since it includes also trumptweets
#  Question: Do we need to use same model bpe_subword for generating both dataset as well?


cat data/trump_tweets_questions.txt data/corrnel_movies_questions.txt >  data/trump_corrnel_questions.txt
cat data/trump_tweets_answers.txt data/corrnel_movies_answers.txt >  data/trump_corrnel_answers.txt

questions=data/trump_corrnel_questions.txt
answers=data/trump_corrnel_answers.txt
# outputs
questions_bpe=data/trump_corrnel_questions.bpe
answers_bpe=data/trump_corrnel_answers.bpe
selected_voc=data/trump_corrnel_selected_voc.txt
words2ids=data/trump_corrnel_words2ids.txt
wordid_questions=data/trump_corrnel_wordid_questions.txt
wordid_answers=data/trump_corrnel_wordid_answers.txt
dict=data/trump_corrnel_questions_en_bpe.dict
prefix=trump_corrnel_qa_word
# params
T=500

# create bpe
bash create_bpe_subword.sh $questions $answers $questions_bpe $answers_bpe $selected_voc $T

# create en-bpe dict
bash create_english_bpe_dict.sh $questions_bpe $dict

# create wordid versions
python qa_data_preprocessing.py $questions_bpe $answers_bpe $selected_voc $words2ids $wordid_questions $wordid_answers


# create tfrecords
python qa_data_prep.py $prefix $words2ids $wordid_questions $wordid_answers



# only trump data

# inputs
questions=data/trump_tweets_questions.txt
answers=data/trump_tweets_answers.txt
# outputs
questions_bpe=data/trump_tweets_questions.bpe
answers_bpe=data/trump_tweets_answers.bpe
#selected_voc=data/trump_tweets_selected_voc.txt
words2ids=data/trump_tweets_words2ids.txt
wordid_questions=data/trump_tweets_wordid_questions.txt
wordid_answers=data/trump_tweets_wordid_answers.txt
dict=data/trump_tweet_questions_en_bpe.dict
prefix=trump_tweet_qa_word
# params
T=500

# apply the bpe model
bash apply_bpe.sh  $questions $answers $questions_bpe $answers_bpe $T

# create wordid versions
python qa_data_preprocessing.py $questions_bpe $answers_bpe $selected_voc $words2ids $wordid_questions $wordid_answers


# create tfrecords
python qa_data_prep.py $prefix $words2ids $wordid_questions $wordid_answers


