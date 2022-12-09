import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns


def get_cleaned_data():
  df = pd.read_csv('spam.csv', delimiter=',')

  df.drop(df.columns.difference(['v1', 'v2']), 1, inplace=True)

  nRow, nCol = df.shape
  print(f'There are {nRow} rows and {nCol} columns')

  df = df.rename({
    "v1": "Category",
    "v2": "Message"
  }, axis=1)

  df["Label"] = df["Category"].map({
    "ham": 0,
    "spam": 1,
  })

  print(df.head(10))
  # check for null values
  print(df.isnull().sum())

  # delete them
  df = df.dropna()
  print(f'There are {df.shape[0]} rows and {df.shape[1]} columns')

  # print value count
  print(df["Category"].value_counts())

  # plot data points, we can see its very imbalanced
  sns.countplot(data=df, x="Category")
  plt.title("ham vs spam")
  plt.show()

  # preproccessing
  ps = PorterStemmer()

  def clean_data():
    corpus = []

    for msg in df["Message"]:
      # replace everything thats not a letter with white space
      msg = re.sub("[^a-zA-Z]", " ", msg)

      # replace new line with whitespace
      msg = msg.replace('\n', '')

      # convert to lowercase
      msg = msg.lower()

      # split the word into individual word list
      msg = msg.split()

      # perform stemming using PorterStemmer for all non-english-stopwords
      msg = [ps.stem(words)
             for words in msg
             if words not in set(stopwords.words("english"))
             ]
      # join the word lists with the whitespace
      msg = " ".join(msg)

      corpus.append(msg)
    return corpus

  data = {'Message': clean_data(), 'Label': df.Label}
  df_new = pd.DataFrame(data=data)

  # new df head
  print(df_new.head())

  # split our data from training & testing
  X_train, X_test, y_train, y_test = train_test_split(df_new.Message, df_new.Label, test_size=0.2, random_state=0)

  print(f'X_train: {X_train.shape}')


  # tokenize our data
  vocab_size = 10000  # only consider this many words

  tokenizer = Tokenizer(num_words=vocab_size)
  tokenizer.fit_on_texts(X_train)
  X_train = np.array(tokenizer.texts_to_sequences(X_train))
  X_test = np.array(tokenizer.texts_to_sequences(X_test))

  # add padding to make tokens equal size
  sentence_len = 200

  X_train = pad_sequences(X_train, maxlen=sentence_len)
  X_test = pad_sequences(X_test, maxlen=sentence_len)



  # Create a SMOTE object
  smote = SMOTE(random_state=42)

  # Transform the data using the SMOTE object
  X_train, y_train = smote.fit_resample(X_train, y_train)

  # Count the number of spam and ham samples
  # Import the seaborn and matplotlib.pyplot libraries

  # Count the number of spam and ham samples
  class_counts = Counter(y_train)

  # Create a bar plot of the class counts
  ax = sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), color="blue")

  # Set the x-axis labels
  ax.set_xticklabels(["Ham", "Spam"])

  # Set the y-axis label
  ax.set_ylabel("Count")

  # Set the plot title
  ax.set_title("Spam vs. Ham Counts")

  # Show the plot
  plt.show()


  return X_train, X_test, y_train, y_test, vocab_size, sentence_len
