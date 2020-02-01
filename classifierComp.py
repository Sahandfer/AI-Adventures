from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

iris = load_iris() # Load the iris dataset
X= iris.data
Y= iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

# using decision tree classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# using k nearest neighbors
classifier1 = KNeighborsClassifier()
classifier1.fit(x_train, y_train)

predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions))

predictions1 = classifier.predict(x_test)
print(accuracy_score(y_test, predictions1))