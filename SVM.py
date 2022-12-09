import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam.csv', delimiter=',', encoding='latin-1')

df.head(n=10)

X = df['Text'].values
y = df['Label'].values

# pre-processing stop words
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def pre_process(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(ps.stem(w))
    return " ".join(filtered_sentence)

X = [pre_process(text) for text in X]

# vectorization
# TF-IDF (Term Frequency - Inverse Document Frequency)
# TF = (Number of times term t appears in a document)/(Number of terms in the document)
# IDF = log_e(Total number of documents/Number of documents with term t in it)
# TF-IDF = TF * IDF
# TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# split our data from training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train our model
from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# test our model
y_pred = clf.predict(X_test)

# evaluate our model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

# test our model with new data

new_data = ["Congrats!, you've just won 1,000$ Amazon Gift Card please click on the link here: http://bit.ly/123456", "Hey, you forgot your 20$ at my place!"]
new_data = [pre_process(text) for text in new_data]
new_data = vectorizer.transform(new_data)
new_data_pred = clf.predict(new_data)
print(new_data_pred)

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

ax = plt.subplot()
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, ax = ax,cmap='Blues',fmt='')

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted labels')
ax.xaxis.set_ticklabels(['Not Spam', 'Spam'])
ax.set_ylabel('True labels')
ax.yaxis.set_ticklabels(['Not Spam', 'Spam'])

