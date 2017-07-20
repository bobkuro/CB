import numpy as np
import sklearn.datasets
import sklearn.svm

from sklearn import svm, datasets

clf = sklearn.svm.SVC()
iris = datasets.load_digits()
ids = np.arange(len(iris["target"]))
P=120
np.random.shuffle(ids)
tr_ids = ids[:P]
te_ids = ids[P:]

X = iris.data[tr_ids]
y = iris.target[tr_ids]

#P = 100
#ids = np.arange(len(iris["target"]))
#np.random.shuffle(ids)
#t_ids = ids[:P]
#e_ids = ids[P:]
clf.fit(X,y)

#clf.fit(iris["data"][t_ids], iris["target"][t_ids])

# Evaluate classifier
print("Predict: ", end = "")
print(clf.predict(iris["data"][te_ids]))
print("Actual:  ", end = "")
print(iris["target"][te_ids])
print("Score:   ", end = "")
print(clf.score(iris["data"][te_ids], iris["target"][te_ids]))

