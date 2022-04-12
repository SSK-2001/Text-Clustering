from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
'''
 Saikrishna  S 19BBS0086
 Text Clustering Class Activity
'''
''' Reading the file and then converting content into list'''
file = open('Trump Response to Healthcare Bill Failure.txt', 'r')
documents = file.readlines()
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
# defining tokens the count of clusters = 2
token = 2
model = KMeans(n_clusters=token, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(token):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()
print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)
