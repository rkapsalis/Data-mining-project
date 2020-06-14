import pandas as pd
import nltk
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import EarlyStopping

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

df = pd.read_csv(r'onion-or-not.csv',encoding='utf8')
pd.options.display.width = None

df.head(10)

df["tokens1"] = df["text"].apply(nltk.word_tokenize) #with punctuation
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df["text"].apply(tokenizer.tokenize) #without punctuation

df.head(10)

stemmer = PorterStemmer()
df['stemmed'] = df['tokens'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
#df = df.drop(columns=['unstemmed']) # Get rid of the unstemmed column.
df.head(10) # Print dataframe.

df['without_stopwords'] = df['stemmed'].apply(lambda x: [item.lower() for item in x if item.lower() not in stop])
df.head(10)

tf = TfidfVectorizer(lowercase='false', analyzer='word')
st = df['without_stopwords'].apply(lambda x: ' '.join(x))

response = tf.fit_transform(st)
re = np.array(response)