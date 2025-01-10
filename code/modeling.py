
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



# Read the processed dataset
df = pd.read_csv("../data/final_data.csv")
print(df.head())

df['Home Win/Loss'] = df['Home Win/Loss'].map({'W': 'Win', 'L': 'Loss'})

# Create X and Y Axes
X = df.drop(['Home Win/Loss'], axis=1)
Y = df['Home Win/Loss']

# Normalize the Axes
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=42)

# Set seed for same results
seed = 0

# Declare the models
knn = KNeighborsClassifier()
gnb = GaussianNB()
mnb = MultinomialNB()
cart = DecisionTreeClassifier(random_state=seed)
rf = RandomForestClassifier(random_state=seed)
svm = SVC(random_state=seed)
# supposedly needs standard scaler instead of minmax?
lr = LogisticRegression(random_state=seed)


# K Nearest Neighbors
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nK Nearest Neighbors Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nK Nearest Neighbors Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nK Nearest Neighbors Classification Report:\n", classReport)


# Multinomial Naive Bayes
mnb.fit(X_train, Y_train)
Y_pred = mnb.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nMultinomial Naive Bayes Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nMultinomial Naive Bayes Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nMultinomial Naive Bayes Classification Report:\n", classReport)


# Gaussian Naive Bayes
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nGaussian Naive Bayes Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nGaussian Naive Bayes Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nGaussian Naive Bayes Classification Report:\n", classReport)


# CART
cart.fit(X_train, Y_train)
Y_pred = cart.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nCART Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nCART Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nCART Classification Report:\n", classReport)


# SVM
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nSVM Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nSVM Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nSVM Classification Report:\n", classReport)


# Random Forest
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nRandom Forest Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nRandom Forest Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nRandom Forest Classification Report:\n", classReport)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
# Number of features to consider at every split
max_features = ['sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_distributions = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
print(param_distributions)

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, Y_train)
rf_random.best_params_

print(rf_random.best_params_)

# {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1,
#  'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
rf = RandomForestClassifier(random_state=seed, n_estimators=500, min_samples_split=5,
                            min_samples_leaf=1, max_features='sqrt', max_depth=10, bootstrap=True)

# Random Forest Tree After Hyper Parameter Tuning 
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nRandom Forest After Hyper Parameter Tuning Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nRandom Forest After Hyper Parameter Tuning Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nRandom Forest After Hyper Parameter Tuning Classification Report:\n", classReport)


# Normalize the Axes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Train and Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=42)

# Logistic Regression
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

accuracy_score(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred) * 100
print("\nLogistic Regression Accuracy:", accuracy)

confMatrix = confusion_matrix(Y_test, Y_pred)
print("\nLogistic Regression Confusion Matrix:\n", confMatrix)

classReport = classification_report(Y_test, Y_pred)
print("\nLogistic Regression Classification Report:\n", classReport)

# Important Features
important_features_dict = {}
for x, i in enumerate(rf.feature_importances_):
    important_features_dict[x] = i


important_features_list = sorted(important_features_dict,
                                 key=important_features_dict.get,
                                 reverse=True)

print('Most important features:\n %s' % important_features_list)
important_features_dict


#Features from Rank 1 to end
cols = list(X.columns)
final_features = []
for i in important_features_list:
  final_features.append(cols[i])
final_features

#Plot the importances
importances = rf.feature_importances_
indices = np.argsort(importances)
indices = indices[::-1]
cols = list(X.columns)
plt.figure(figsize=(20,18))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#85baff', align='center',height=0.5)
plt.yticks(range(len(indices)), final_features, rotation ='horizontal')
plt.ylabel('Relative Importance')
plt.xlabel('Significance')
plt.show()
