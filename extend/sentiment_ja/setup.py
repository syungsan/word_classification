from setuptools import setup, find_packages

setup(name="sentimentja",
      version="2.1.17",
      description="Sentiment Analysis for Japanese",
      author="Shun Sugiyama",
      url="https://github.com/sugiyamath/sentiment_ja",
      packages=['sentimentja', 'sentimentja.src'],
      install_requires=[
          "tensorflow",
          "youtokentome"
      ],
      package_data={
          'sentimentja':["*"],
          'sentimentja.src':["*"],
      }
)
