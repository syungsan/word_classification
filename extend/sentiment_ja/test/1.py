import unittest
from sentimentja import Analyzer, sentiment
from pprint import pprint

analyzer = Analyzer()
sents = [
    "final fantasy 14 超楽しい",
    "クソゲーはつまらん",
    "エアリスが死んで悲しい",
    "冒険の書が消える音こわい",
    "廃人ゲーマーのスキルすごい",
    "ケフカキモい",
    "コロナ"
]
probs = analyzer(sents)
values = probs[0]
ktype1 = {"sentence", "emotions"}
ktype2 = {"happy", "sad", "angry", "disgust", "surprise", "fear"}


class TestAnalyzer(unittest.TestCase):
    def test_key(self):
        self.assertEqual(ktype1, set(values.keys()))

    def test_key2(self):
        self.assertEqual(ktype2, set(values["emotions"].keys()))
    
    def test_type_sentence(self):
        self.assertEqual(str, type(values["sentence"]))

    def test_type_emotions(self):
        self.assertEqual(dict, type(values["emotions"]))

    def test_type_each_emotions(self):
        for v in values["emotions"].values():
            self.assertEqual(float, type(v))

    def test_range(self):
        for _, v in values["emotions"].items():
            self.assertTrue(0 <= v and v <= 1)

    def test_num_of_emotions(self):
        self.assertEqual(6, len(values["emotions"].keys()))

if __name__ == '__main__':
    pprint(probs)
    unittest.main()

