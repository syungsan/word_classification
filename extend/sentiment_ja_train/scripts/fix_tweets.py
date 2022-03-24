import pandas as pd
import random


def fix(df):

    min_size = min(df[df["label"] == i].shape[0] for i in range(6))
    dfs = []
    for i in range(6):
        tdf = df[df["label"] == i][["tweet", "label"]]
        tdf = tdf.iloc[random.sample(range(0, tdf.shape[0]), k=min_size)]
        dfs.append(tdf)
    dfs = pd.concat(dfs)
    dfs.to_csv("../data/tweets_fixed.csv", encoding='utf-8')


if __name__ == "__main__":
    df = pd.read_csv("../data/tweets.csv", encoding='utf-8')
    fix(df)
