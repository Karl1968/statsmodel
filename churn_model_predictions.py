#######################################
###  Diviser mon dataset en training et Test
#######################################

### Tranformer mes Variables en Matrices
X1=X.values
y1=y.values

# Diviser le dataset entre le Training set et 10% Test set
def set_train_test(xTrain,yTrain,percent):
    from sklearn.model_selection import train_test_split
    global X_train, X_test, y_train, y_test
    #response = input("Veuillez saisir la valeur pour le test set:  ")
    #percent=int(response)
    percent=int(percent)
    X_train, X_test, y_train, y_test = train_test_split(xTrain, yTrain, test_size = percent/100, random_state = 0)

set_train_test(X1,y1,10)

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
