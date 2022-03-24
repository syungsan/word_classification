import twint
import threading
import time


def scraping(search, file):

    c = twint.Config()

    c.Search = search
    c.Limit = 1000000  # Not working.
    c.Store_csv = True
    c.Lang = "ja"
    c.Output = file

    twint.run.Search(c)


if __name__ == "__main__":

    files = ("../data/{}".format(x) for x in (
        "happy.csv", "sad.csv", "angry.csv",
        "disgust.csv", "surprise.csv", "fear.csv"
    ))

    searches = ["😄", "😢", "😲", "🤮", "😡", "😨"]

    threadlist = []
    for index, file in enumerate(files):

        sc = threading.Thread(target=scraping, args=(searches[index], file))
        sc.start()
        threadlist.append(sc)
        time.sleep(5)

    for th in threadlist:
        th.join()
