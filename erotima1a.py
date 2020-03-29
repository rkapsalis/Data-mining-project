import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
data = pd.read_csv (r'winequality-red.csv')
print(data.head())
print(data.shape)
print(data.describe())
print(data.quality.value_counts())

#categorize wine quality
# 1–4  poor quality, 5–6 average, 7–10  high
bins = [1, 6, 10]

#0 = low quality, 1 = high
quality_labels = [0, 1]
data['quality_class'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)

#Displays the first 2 columns
print(data.head(n=2))
print(data['quality_class'].value_counts())

y = data['quality_class']
X = data.drop(['quality', 'quality_class'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sc = StandardScaler()  #scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train.head())
print(X_train.shape)

#print(X_test.head())
print(X_test.shape)

classifier = SVC(kernel='rbf', random_state=0, gamma = 0.9)
trained_model = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
print(accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test, y_pred))
