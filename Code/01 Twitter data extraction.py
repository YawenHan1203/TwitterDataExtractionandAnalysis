
# load libraries
import tweepy


#Variables that contains the user credentials to access Twitter API
#Twitter credentials for the app
# need to use you account tokens and keys: https://developer.twitter.com/
access_token = "XXX"
access_token_secrete = "XXX"
consumer_key = "XXX"
consumer_secrete = "XXX"


 
#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secrete)
auth.set_access_token(access_token, access_token_secrete)
api = tweepy.API(auth,wait_on_rate_limit=True)


#declare keywords as a query for three categories
btc_keywords = '#bitcoin OR #btc'

import json


list=[]
with open('test.json', 'w', encoding='utf-8') as f:
    tweet = tweepy.Cursor(api.search, q=btc_keywords+" -filter:retweets", tweet_mode='extended',
                              count=200, include_rts=False, since='2019-08-29', until='2019-08-30').items(5)
    for t in tweet:
        list.append(t._json)
    
    json.dump(list, f, ensure_ascii=False, indent=4)





