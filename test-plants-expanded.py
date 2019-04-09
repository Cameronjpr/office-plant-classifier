# Cactus, daffodil, or baby palm? 
from sklearn import tree

# training data below - used later to train the classifier to distinguish between the two kinds of plants
# first value is height in cm 
# second value is has leaves (0 false, 1 true)
# third value is has petals (0 false, 1 true)
# fourth value is has needles (0 false, 1 true)
features = [[100, 1, 0, 0], [15, 0, 0, 1], [40, 0, 1, 0], [110, 1, 0, 0], [18, 0, 0, 1], [35, 0, 1, 0]]
labels = [2, 0, 1, 2, 0, 1] # catus 0, daff 1, baby palm 2

# classifier 
# set clf to be a Decision Tree Classifier
# fit tries to find patterns in the training data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# predict using knowledge gained from calling fit on the training data
print(clf.predict([[20, 0, 0, 1]]))
