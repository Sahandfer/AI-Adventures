
from sklearn import tree
features = [[150, 1],[150, 2], [100, 3], [100, 2], [120, 1] ] # The adjectives 
labels = ['orange', 'apple', 'banana', 'apple', 'orange']  # The desired answers
classifier = tree.DecisionTreeClassifier() # Make a classifier tree
clf = classifier.fit(features, labels) # Make the rules based on training data and desired answers
result = clf.predict([[140,1]]) # Test the classifier
print (result[0]) 