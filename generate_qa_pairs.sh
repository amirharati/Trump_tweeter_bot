# generate_qa_pairs.sh
# Amir Harati, Aug 2018
# create pais of question and answers using only tweets.
# idea is to extarct different things from a tweet and using it as question.
python generate_qa_pairs.py  data/freq_grams.list data/trump_tweets_text_data.txt  data/trump_tweets_questions.txt  data/trump_tweets_answers.txt
