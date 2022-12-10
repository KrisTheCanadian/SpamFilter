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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Data Preprocessing
def get_cleaned_data():
  df = pd.read_csv('spam.csv', delimiter=',')

  df.drop(df.columns.difference(['v1', 'v2']), 1, inplace=True)

  nRow, nCol = df.shape

  df = df.rename({
        "v1": "Category",
        "v2": "Message"
      }, axis=1)

  df["Label"] = df["Category"].map({
        "ham": 0,
        "spam": 1,
      })

  # delete null values and delete duplicates 
  df = df.dropna()
  df1 = df.loc[df['Label'] == 0].drop_duplicates()
    
  df = pd.concat([df1,df.loc[df['Label'] == 1]])
  

#   # Appending 2nd dataset
  df2 = pd.read_csv('spamExtra.csv', delimiter=',')

  df2.drop(df2.columns.difference(['v1', 'v2']), 1, inplace=True)

  nRow, nCol = df2.shape

  df2 = df2.rename({
        "v1": "Category",
        "v2": "Message"
      }, axis=1)

  df2["Label"] = df2["Category"].map({
        "ham": 0,
        "spam": 1,
      })

  # delete them
  df2 = df2.dropna()

  # Select rows belonging to class 1 from df2
  df2_class_1 = df2.loc[df2['Label'] == 1]
  df = pd.concat([df, df2_class_1])
# +++++++++++++++++++++++++++++++++++++++++++++++++

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
  X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
  X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

  # add padding to make tokens equal size
  sentence_len = 200

  X_train = pad_sequences(X_train, maxlen=sentence_len)
  X_test = pad_sequences(X_test, maxlen=sentence_len)



  # Create a SMOTE object
  smote = SMOTE(random_state=42)

  # Transform the data using the SMOTE object
  X_train, y_train = smote.fit_resample(X_train, y_train)

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

# Helper Functions
def generate_model_output(y_test, y_pred):
    ax = plt.subplot()
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax = ax,cmap='Blues',fmt=''); 

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('Predicted labels');
    ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); 
    ax.set_ylabel('True labels');
    ax.yaxis.set_ticklabels(['Not Spam', 'Spam']);

    print("Classification Report\n", classification_report(y_test, y_pred), "\n")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))