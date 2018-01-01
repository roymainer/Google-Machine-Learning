import numpy as np
from builtins import print
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
test_indexes = [0, 50, 100]  # the indexes of the first of each iris type

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])

# print the data
# for i in range(len(iris.target)):
#     print("Example {0}: label {1}, features {2}".format(i, iris.target[i], iris.data[i]))

"""
we need to remove some of the data to use as testers instead of trainers
"""

# training data
train_target = np.delete(iris.target, test_indexes)
train_data = np.delete(iris.data, test_indexes, axis=0)

# testing data
test_target = iris.target[test_indexes]
test_data = iris.data[test_indexes]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# visualize the code
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
