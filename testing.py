import pandas as pd
import numpy as np
import math as math

#wine = pd.read_csv("winequality-red.csv", sep=";")
#wine.head

from sklearn import linear_model
clf = linear_model.LinearRegression()

# (x+1)^2
#if X > (61-10*(math.sqrt(7)))/29
#Y =
#if X = otherwise
#Y = 20*(x-3)^2-20
X = [[-4,9] , [-1,0] , [1.8180,7.9413] , [3,-20] , [4.9,52.2]]
Y = [1,2,3,4,5]
# 予測モデルを作成
clf.fit(X, Y)

# 回帰係数
print(clf.coef_)


# 決定係数
print(clf.score(X, Y))

