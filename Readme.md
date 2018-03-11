A Trump tweeter bot with conditional words. So the bot generate a tweet with symantically similar word/words as input word (or words)in it.
The idea is a simple conditional LM with an input vector sampled from word2vec table during training (we sample so model does not necessary use exact same word).
During test we can just feed the given word word2vec to the model.
data:
https://github.com/bpb27/trump_tweet_data_archive
