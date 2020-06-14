import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = pd.read_csv (r'winequality-red.csv')
print (data.head())
print (data.shape)
print(data.describe())
print(data.quality.value_counts())
y = data.quality
x = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(X_train.head())
print(X_train.shape)

print(X_test.head())
print(X_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#features_raw = data.drop(['quality'], axis = 1)
classifier = SVC(C=0.01, kernel='poly', degree=3, gamma=0.1, class_weight='balanced')
trained_model = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print(f1_score(y_test, y_pred, average="weighted"))
print(precision_score(y_test, y_pred, average="weighted"))
print(recall_score(y_test, y_pred, average="weighted"))
print(accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = y_train, cv = 10)
#we can see model's average accuracy
print(accuracies.mean())
