import pandas as pd
import nltk
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
nltk.download('punkt')
stop = stopwords.words('english')


df = pd.read_csv(r'onion-or-not.csv', encoding='utf8')
pd.options.display.width = None
print(df.shape)
print(df.head(10))

df["tokens1"] = df["text"].apply(word_tokenize)  # with punctuation
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df["text"].apply(tokenizer.tokenize)  # without punctuation
print(df.head(10))

stemmer = PorterStemmer()
df['stemmed'] = df['tokens'].apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.
df.head(10)  # Print dataframe.

df['without_stopwords'] = df['stemmed'].apply(lambda x: [item.lower() for item in x if item.lower() not in stop])
df.head(10)

tf = TfidfVectorizer(lowercase='false', analyzer='word')
st = df['without_stopwords'].apply(lambda x: ' '.join(x))

re = tf.fit_transform(st)
response = tf.fit_transform(st).toarray()
print(re.shape)
print(re)
rows, cols = np.nonzero(response)

print(response[rows, cols])
print(response.shape)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


X = response
Y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print(X_train.shape)
model = Sequential()

model.add(Dense(16, activation='sigmoid', input_dim=16521))
model.add(Dense(16, activation='relu', input_dim=16521))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

# Stop training when a monitored metric has stopped improving
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, verbose=1, mode='min')
history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    batch_size=16,
                    callbacks=[early_stopping],
                    validation_data=(X_test, y_test),
                    verbose=1)

model.summary()
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=1)
