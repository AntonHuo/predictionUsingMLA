import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as kNN


data = pd.read_csv('testdateimitlabels.csv',low_memory=False)


col_names = data.columns
col_list = col_names.tolist()

keys_X = []
for x in range(5,839):
    keys_X.append(col_list[x])

X = data[keys_X]

X = X.select_dtypes(exclude=['bool','object'])

imp = SimpleImputer(strategy="mean")

X = imp.fit_transform(X)

X = pd.DataFrame(X)
print(X)

X.info()

y1 = data[['all']]
y2 = data[['relevant']]
y3 = data[['relevant5%']]

########################################################################################################################

y1.info()
print(y1)

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=0)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation
param_dist = {
        'n_neighbors':range(1,30),
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
grid.fit(X_train, y_train.values.ravel())

best_estimator = grid.best_estimator_
print(best_estimator)

#nach cross validation bekommen wir best_estimator.
clf = best_estimator

print('the acuracy for all is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

########################################################################################################################

y2.info()
print(y2)

X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation
param_dist = {
        'n_neighbors':range(1,30),
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
grid.fit(X_train, y_train.values.ravel())

best_estimator = grid.best_estimator_
print(best_estimator)

clf = best_estimator

print('the acuracy for relevant is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

########################################################################################################################

y3.info()
print(y3)

X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.2, random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#cross validation
param_dist = {
        'n_neighbors':range(1,30),
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
grid.fit(X_train, y_train.values.ravel())

best_estimator = grid.best_estimator_
print(best_estimator)

clf = best_estimator

print('the acuracy for relevant5% is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))