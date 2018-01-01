"""
Google first machine learning video:
https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLT6elRN3Aer7ncFlaCz8Zz-4B5cnsrOMt

This is a basic classifier for Oragnes and Apples
"""

from sklearn import tree

ORANGE = 1
APPLE = 0

features = [[140,1], [130, 1], [150, 0], [170, 0]]
labels = ["apple", "apple", "orange", "orange"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))