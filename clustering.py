from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


#a string array that contains our document
document = ["This is the most beautiful place in the world.", "This man has more skills to show in cricket than any other game.", "Hi there! how was you trip last month?",
            "There was a player who scored 200+ runs in single cricket innings in his career.", "I have got the opportunity to travel to Paris next year for my internship.",
            "Maybe he is better than you in batting but you are much better than him in bowling.", "That was really a great day for me when I was there at Lavasa for the whole night.",
            "That's exactly what I wanted to become, the highest ratted batsmen ever with top scores.", "Does it really matter whether you got to Thailand or Goa, its just you have to spend your holidays."
            "Why don't you go to Switzerland next year for your 25th Wedding anniversary?", "Travel is fatal to prejudice, bigotry, and narrow mindedness, and many of our people need it sorely on these accounts.",
            "Stop worrying about the potholes in the road and enjoy the journey.", "No cricket team in the world depends on one or two players. The team always plays to win.",
            "Soccer is a team game. If you want fame for yourself, go play an individual game.", "Because in the end, you won't remember the time you spent working in the office or mowing your lawn. Clime that damn mountain.",
            "Isn't cricket supposed to be a team sport? I feel people should decide first whether cricket is a team game or an individual sport."]


#create a vectorizer class to transform the document, also takes care of the general stop words
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(document)


#k-means is an unsupervised model for clustering
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print(model)


#centroids are essentially mega clusters of important words
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

#here we are just printing the words in the clusters
for i in range(true_k):
    print('Cluster %d:' % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])

#printing output
print("\n")
print("Prediction")
X = vectorizer.transform(["Northing is easy in basketball. Maybe when you watch it on TV, it looks easy. But it is not. You have to use your brain and time the ball."])
predicted = model.predict(X)
print(predicted)
