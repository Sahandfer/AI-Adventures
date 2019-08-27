from sklearn.datasets import load_iris
from sklearn import tree
import numpy

iris = load_iris() # Load the iris dataset

temp = [0,50,100]

# Data for training
target_TR = numpy.delete(iris.target, temp)
data_TR = numpy.delete(iris.target, temp, axis = 0)

# Data for testing
target_TS = iris.target[temp]
data_TS = iris.data[temp]

classifier = tree.DecisionTreeClassifier()
clf = classifier.fit(target_TR, data_TR)

result = clf.predict(data_TS)
print(result)