# sentimentja
Sentiment Analyzer for Japanese Language

- all versions: https://github.com/sugiyamath/sentiment_ja/tags
- train your own model: https://github.com/sugiyamath/sentiment_ja_train

## Installation

You can use ```make``` command.

```
git clone https://github.com/sugiyamath/sentiment_ja
cd sentiment_ja
make
```

or just simply this:

```
git clone https://github.com/sugiyamath/sentiment_ja
cd sentiment_ja
python setup.py install
```

## Example usage

```python
from sentimentja import Analyzer
from pprint import pprint

analyzer = Analyzer()
pprint(analyzer.analyze([
    "final fantasy 14 超楽しい",
    "クソゲーはつまらん",
    "エアリスが死んで悲しい",
    "冒険の書が消える音こわい",
    "廃人ゲーマーのスキルすごい",
    "ケフカキモい",
    "コロナ"
]))
```

[result]

```python
[{'emotions': {'angry': 0.113675155,
               'disgust': 0.115149096,
               'fear': 0.10165207,
               'happy': 0.7016944,
               'sad': 0.2467626,
               'surprise': 0.13182622},
  'sentence': 'final fantasy 14 超楽しい'},
 {'emotions': {'angry': 0.31707087,
               'disgust': 0.1622563,
               'fear': 0.14181371,
               'happy': 0.23686178,
               'sad': 0.36883706,
               'surprise': 0.1088921},
  'sentence': 'クソゲーはつまらん'},
 {'emotions': {'angry': 0.10301053,
               'disgust': 0.09832993,
               'fear': 0.111722335,
               'happy': 0.09970104,
               'sad': 0.8859446,
               'surprise': 0.102533355},
  'sentence': 'エアリスが死んで悲しい'},
 {'emotions': {'angry': 0.09739415,
               'disgust': 0.13288762,
               'fear': 0.6327518,
               'happy': 0.096388154,
               'sad': 0.3440729,
               'surprise': 0.09998626},
  'sentence': '冒険の書が消える音こわい'},
 {'emotions': {'angry': 0.10138765,
               'disgust': 0.14274225,
               'fear': 0.33686018,
               'happy': 0.15706311,
               'sad': 0.1737709,
               'surprise': 0.49469906},
  'sentence': '廃人ゲーマーのスキルすごい'},
 {'emotions': {'angry': 0.114263475,
               'disgust': 0.6433052,
               'fear': 0.20415862,
               'happy': 0.11632563,
               'sad': 0.1609963,
               'surprise': 0.096255496},
  'sentence': 'ケフカキモい'},
 {'emotions': {'angry': 0.15206008,
               'disgust': 0.1143392,
               'fear': 0.15766238,
               'happy': 0.26258203,
               'sad': 0.43910575,
               'surprise': 0.14823207},
  'sentence': 'コロナ'}]
```
