# download the data (and remove old data)
mkdir raw_data
cd raw_data
#
for year in 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018
do
   wget "https://raw.github.com/bpb27/trump_tweet_data_archive/master/condensed_$year.json.zip"
   wget https://raw.github.com/bpb27/trump_tweet_data_archive/master/master_$year.json.zip
done

unzip "*.zip"

