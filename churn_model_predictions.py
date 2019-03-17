#######################################
###  Diviser mon dataset en training et Test
#######################################

### Tranformer mes Variables en Matrices
X1=X.values
y1=y.values

# Diviser le dataset entre le Training set et 10% Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.10, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Construction du Modéle
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


### Faire de nouvelles Predictions
y_pred = classifier.predict(X_train)

y_pred

### Matrice de Confusion pour le test Set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

cm
