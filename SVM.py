import numpy as np
import sklearn.datasets
import sklearn.svm

# Load dataset
dataset = sklearn.datasets.load_iris()

# Set training set and evaluation set
P = 149
ids = np.arange(len(dataset["target"]))
np.random.shuffle(ids)
t_ids = ids[:P]
e_ids = ids[P:]

# Train classifier
classifier = sklearn.svm.SVC()
classifier.fit(dataset["data"][t_ids], dataset["target"][t_ids])

# Evaluate classifier
print("Predict: ", end = "")
print(classifier.predict(dataset["data"][e_ids]))
print("Actual:  ", end = "")
print(dataset["target"][e_ids])
print("Score:   ", end = "")
print(classifier.score(dataset["data"][e_ids], dataset["target"][e_ids]))
