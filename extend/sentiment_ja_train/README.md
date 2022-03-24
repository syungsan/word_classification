
# sentiment_ja_train
Training pipeline of sentiment_ja: https://github.com/sugiyamath/sentiment_ja


## How sentiment_ja works?

1. Search Tweets by using emojis as query. That emojis are corresponding to target emotions.
2. Scrape tweets. You can use twitterscraper: https://github.com/taspinar/twitterscraper
3. Annotate tweets by emojis. To prevent data leakage, remove emojis from texts. 
4. Build a model by using keras. Input: BPE encoded tweets, Output: emotion label
5. Test.

## Workflow

The complete workflow is defined as Makefile. You just need to run:

```
make
```

Warning: I hadn't tested Makefile yet.
