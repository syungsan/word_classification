import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    df = pd.read_csv("../data/tweets.csv", encoding='utf-8')["tweet"]
    with open("../data/tweet_texts.txt", "w", encoding="utf-8") as f:
        for text in tqdm(df):
            text = str(text).replace("\n", " ")
            f.write(text + "\n")
