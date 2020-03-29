import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv (r'winequality-red.csv')
print (data.head())
print (data.shape)
print(data.describe())
print(data.quality.value_counts())

#categorize wine quality
# 1–4  poor quality, 5–6 average, 7–10  high
bins = [1,4,6,10]

#0 = low quality, 1 = average, 2 = high
quality_labels=[0,1,2]
data['quality_class'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)

#Displays the first 2 columns
print(data.head(n=2))
print(data['quality_class'].value_counts())

y = data['quality_class']
x = data.drop(['quality', 'quality_class'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(x_train.head())
print(x_train.shape)

print(x_test.head())
print(x_test.shape)

classifier = SVC(kernel = 'linear', random_state = 0)
trained_model=classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

#Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
