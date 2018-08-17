# create_trumptweet_dataset_pipeline.sh
# Amir Harati Aug 2018
# This a pipeline to create the input of chatbot_dataprepration pipeline from trump tweets

# creat input text from raw tweets.
python rawtrump_data_preprocessing.py

# find freq n-grams  save in data/freq_grams.list
bash find_freq.sh

# generate question/answers pairs for trump tweets
# outputs: data/trump_tweets_questions.txt  data/trump_tweets_answers.txt 
bash generate_qa_pairs.sh

