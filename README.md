Work under prepration.

A Trump tweeter bot with conditional words. So the bot generate a tweet with symantically similar word/words as input word (or words)in it.
The idea is a simple conditional LM with an input vector sampled from word2vec table (basdd on distance) during training (we sample so model does not necessary use exact same word).
During test we can just feed the given word vector  to the model.

data:
https://github.com/bpb27/trump_tweet_data_archive

https://github.com/ryanmcdermott/trump-speeches

https://github.com/paigecm/2016-campaign

https://github.com/andrewts129/transcript-scraping

https://github.com/rtlee9/Trump-bot/tree/master/data/trump/speeches
