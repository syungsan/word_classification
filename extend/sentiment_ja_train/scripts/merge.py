import pandas as pd
import demoji
import re


if __name__ == "__main__":

    files = ("../data/{}".format(x) for x in (
        "happy.csv", "sad.csv", "angry.csv",
        "disgust.csv", "surprise.csv", "fear.csv"
    ))

    dfs = []
    for label, fname in enumerate(files):

        df = pd.read_csv(fname, sep=",", encoding='utf-8')
        df["label"] = label

        """
        for index, tweet in enumerate(df["tweet"]):

            tweet = demoji.replace(string=tweet, repl="")
            tweet = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)", "", tweet)

            to_remove = []
            tws = tweet.split(" ")

            for tw in tws:
                tw = re.sub(r"\s", "", tw)

                if "@" in tw or tw == "":
                    to_remove.append(tw)

            _tws = [i for i in tws if i not in to_remove]
            tweet = " ".join(_tws)

            if tweet != "":
                df.at[df.index[index], "tweet"] = tweet
            else:
                df.drop(index=index)
        """

        dfs.append(df)

    dfs = pd.concat(dfs)
    dfs.to_csv("../data/tweets.csv", encoding='utf-8')
