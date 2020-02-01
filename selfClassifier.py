from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

class myClassifier():
    def fit (self, x, y):
        self.x_train = x
        self.y_train = y

    def closest (self, row):
        minDist = distance.euclidean(row, self.x_train[0])
        minIdx = 0
        for i in range(1, len(self.x_train)):
            tempDist = distance.euclidean(row, self.x_train[i])
            if (tempDist<minDist):
                minDist= tempDist
                minIdx = i
        return self.y_train[minIdx]

    def predict(self, x):
        predictions = []
        for row in x:
            label = self.closest(row)
            predictions.append(label)
        return predictions


iris = load_iris() # Load the iris dataset
X= iris.data
Y= iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

# using k nearest neighbors
classifier = myClassifier()
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions))
