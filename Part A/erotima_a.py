import numpy
import pandas
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, \
    classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv(r'winequality-red.csv')  # read file
print(data.quality.describe())
print(data.quality.value_counts())


def plot_wine_ph_histogram(pH):
    unique_vals = data['pH'].sort_values().unique()
    plt.xlabel("pH")
    plt.ylabel("Count")
    plt.hist(pH.values, bins=np.append(unique_vals, 9), align='left')


plot_wine_ph_histogram(data['pH'])


def plot_wine_quality_histogram(quality):  # quality histogram
    unique_vals = data['quality'].sort_values().unique()
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.hist(quality.values, bins=np.append(unique_vals, 9), align='left')


plot_wine_quality_histogram(data['quality'])

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidth=0.5, center=0, cmap='coolwarm')
plt.savefig("quality-ph.png")
plt.show()

y = data.quality
x = data.drop('quality', axis=1)
X_train1, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
sth = X_train1.head()
sth1 = X_test.head()

tuned_parameters = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['linear'],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['sigmoid'],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    # {
    #     'kernel': ['poly'],
    #     'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
    #     'C': [1, 10, 100, 1000],
    #     'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # }
]

sc = StandardScaler()
X_train = sc.fit_transform(X_train1)
X_test = sc.transform(X_test)
classifier = SVC(random_state=5, kernel="rbf", C=10, gamma=0.1)
trained_model = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# clf = GridSearchCV(sklearn.svm.SVC(), tuned_parameters, scoring='accuracy', verbose=10)
# clf.fit(X_train, y_train)
# print(clf.best_estimator_)
# y_pred = clf.best_estimator_.predict(X_test)
# print(sklearn.metrics.classification_report(y_test, y_pred))
# print(clf.cv_results_)
# print(clf.best_estimator_)
# y_pred = clf.best_estimator_.predict(X_test)
# print(sklearn.metrics.classification_report(y_test, y_pred))
# print(clf.cv_results_)

cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)

precision = precision_score(y_test, y_pred, average="weighted")
print(precision)
recall = recall_score(y_test, y_pred, average="weighted")
print(recall)
accuracy = accuracy_score(y_test, y_pred) * 100
print(accuracy)

print(classification_report(y_test, y_pred, digits=4))


def buildSVM(Xtrain, Xtest, ytrain, ytest):
    classifier = SVC(random_state=5, kernel="rbf", C=10, gamma=0.1)
    trained_model = classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xtest)
    print(sklearn.metrics.classification_report(ytest, y_pred, digits=4))


erased = len(X_train) * 33 // 100
not_erased = len(X_train) - erased
nancol = sample(range(len(X_train)), erased)  # παίρνω τυχαίο δείγμα του 33% των rows
for index in nancol:
    X_train[index][8] = None

no_ph = np.delete(X_train, [8], axis=1)  # αφαιρώ τη στήλη ph απο το training
no_ph_test = np.delete(X_test, [8], axis=1)  # αφαιρώ τη στήλη ph απο το test
buildSVM(no_ph, no_ph_test, y_train, y_test)

median = 0
for index in range(len(X_train)):
    if index not in nancol:  # για κάθε γραμμή που δεν περιέχει nan
        median += X_train[index][8]

X_median = numpy.array(X_train)
median = median / not_erased
for index in range(len(X_median)):
    if index in nancol:
        X_median[index][8] = median

buildSVM(X_median, X_test, y_train, y_test)

y_ph = X_train[:, :8]  # η στήλη με τα missing values

xg_train = []
yg_train = []
xg_test = []
yg_test = []
X_train2 = X_train1.values
np.set_printoptions(suppress=True)

X_train2[:, 0] = X_train2[:, 0] * 10
X_train2[:, 1] = X_train2[:, 1] * 1000
X_train2[:, 2] = X_train2[:, 2] * 100
X_train2[:, 3] = X_train2[:, 3] * 10
X_train2[:, 4] = X_train2[:, 4] * 1000
X_train2[:, 7] = X_train2[:, 7] * 100000
X_train2[:, 8] = X_train2[:, 8] * 100
X_train2[:, 9] = X_train2[:, 9] * 100
X_train2[:, 10] = X_train2[:, 10] * 10
X_train2.astype(int)

no_ph = np.delete(X_train2, [8], axis=1)  # χωρίς ph
for i in range(len(X_train2)):
    if i not in nancol:
        xg_train.append(no_ph[i])
        yg_train.append(X_train2[i][8])
    if i in nancol:
        xg_test.append(no_ph[i])
        yg_test.append(X_train2[i][8])

logistic_regression = LogisticRegression(max_iter=120000)
logistic_regression.fit(xg_train, yg_train)
y_pred = logistic_regression.predict(xg_test)

j = 0
for i in range(len(X_train2)):
    if i in nancol:
        X_train2[i][8] = y_pred[j]
        j += 1

X_train2[:, 0] = X_train2[:, 0] / 10
X_train2[:, 1] = X_train2[:, 1] / 1000
X_train2[:, 2] = X_train2[:, 2] / 100
X_train2[:, 3] = X_train2[:, 3] / 10
X_train2[:, 4] = X_train2[:, 4] / 1000
X_train2[:, 7] = X_train2[:, 7] / 100000
X_train2[:, 8] = X_train2[:, 8] / 100
X_train2[:, 9] = X_train2[:, 9] / 100
X_train2[:, 10] = X_train2[:, 10] / 10
X_train2 = sc.fit_transform(X_train2)

buildSVM(X_train2, X_test, y_train, y_test)

no_ph1 = X_train1.drop('pH', axis=1)
X_kmeans = X_train1.values.tolist()
X_kplot = no_ph1.values.tolist()

pH = 8
for index in nancol:
    X_kmeans[index][pH] = None

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_kplot)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("kmeans.png")
plt.show()
# Αυτα ειναι για να δειξουμε γιατι 4


kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
centers = kmeans.fit(X_kplot).cluster_centers_

clusters = []
for i in range(len(centers)):
    clusters.append([])
for i in range(len(X_kplot)):
    dist = []
    for j in range(len(centers)):
        dist.append([np.linalg.norm(X_kplot[i] - centers[j]), j])
    dist.sort()
    clusters[dist[0][1]].append(i)

median = []
for i in range(len(clusters)):
    median.append(0)

clust_num = 0
# print(len(clusters))
for cluster in clusters:
    not_erased = 0
    for index in cluster:
        if X_kmeans[index][pH] is not None:
            median[clust_num] += X_kmeans[index][pH]
            not_erased += 1
    if not_erased != 0:
        median[clust_num] /= not_erased
    for index in cluster:
        if X_kmeans[index][pH] is None:
            X_kmeans[index][pH] = median[clust_num]
    clust_num += 1

X_noph = sc.fit_transform(no_ph)
X_kmeans = sc.fit_transform(X_kmeans)
buildSVM(X_kmeans, X_test, y_train, y_test)
