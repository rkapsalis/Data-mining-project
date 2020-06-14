import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from random import randrange
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv (r'winequality-red.csv')



y = data.quality
x = data.drop('quality', axis=1)
j = x.drop('pH', axis=1)
X_train, X_test ,y_train, y_test ,X_noph ,y_noph = train_test_split(x, y, j, test_size=0.25, random_state=42)

X_median = X_train.values.tolist()
X_kmeans = X_train.values.tolist()
X_kplot = X_noph.values.tolist()

pH=8
erased = len(X_median) * 33 // 100
not_erased = len(X_median) - erased
nancol = random.sample(range(len(X_median)), erased)

median=0
for index in range(len(X_median)):
    if index not in nancol:
        median+=X_median[index][pH] 
median = median/not_erased

for index in nancol:
    X_median[index][pH] = median
    




for index in nancol:
    X_kmeans[index][pH] = None



# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X_kplot)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
#Αυτα ειναι για να δειξουμε γιατι 4


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
centers = kmeans.fit(X_kplot).cluster_centers_

clusters = []
for i in range(len(centers)):
    clusters.append([])
for i in range(len(X_kplot)):
    dist = []
    for j in range(len(centers)):
        dist.append([np.linalg.norm(X_kplot[i]-centers[j]),j])
    dist.sort()
    clusters[dist[0][1]].append(i)

median = []
for i in range(len(clusters)):
    median.append(0)

clust_num = 0
for cluster in clusters:
    not_erased=0
    for index in cluster:
        if X_kmeans[index][pH] is not None:
            median[clust_num] += X_kmeans[index][pH]
            not_erased+=1
    if not_erased != 0:
        median[clust_num] /=not_erased
    for index in cluster:
        if X_kmeans[index][pH] is None:
            X_kmeans[index][pH] = median[clust_num]
    clust_num+=1


sc = StandardScaler()
X_median = sc.fit_transform(X_median)
X_noph = sc.fit_transform(X_noph)
X_kmeans = sc.fit_transform(X_kmeans)


#Για χωρις στηλη εξεταζουμε με X_noph -> y_train για train και με y_noph -> y_test //Μια λιγοτερη είσοδος
#Για τον μέσο όρο εξετάζουμε με X_median -> y_train για train και με X_test -> y_test
#Για τους γειτονες εξετάζουμε με X_kmeans -> y_train για train και με X_test -> y_test