import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

classifier = SVC(kernel="rbf",gamma=0.6,C = 0.5)
classifier.fit(x_train_scaled,y_train)
y_pred = classifier.predict(x_test_scaled)

accuracies = cross_val_score(estimator=classifier,X = x_train_scaled,y = y_train,cv = 10)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


print("Cross-validation accuracies:", accuracies)
print("Mean accuracy:", accuracies.mean())


parameters = [{"C":[0.25,0.5,0.75,1],"kernel":["linear"]},
              {"C":[0.25,0.5,0.75,1],"kernel":["rbf"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},
              {"C":[0.25,0.5,0.75,1],"kernel":["rbf"],"gamma":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],"degree":[1,2,3,4,5,6,7,8,9,10],"coef0":[0,0.1,0.2]}]


grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv = 10,n_jobs=-1)

grid_search.fit(x_train_scaled,y_train)

best_accuracy = grid_search.best_score_
best_parameters  = grid_search.best_params_

print("so for the best accuracy is ",best_accuracy*100,"% and that accuracy was taken from the combo of ",best_parameters)

def plot_decision_boundary(X, y, model, title):
    from matplotlib.colors import ListedColormap
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.legend()
    plt.show()


plot_decision_boundary(x_train_scaled, y_train, classifier, 'SVM (Training set)')

plot_decision_boundary(x_test_scaled, y_test, classifier, 'SVM (Test set)')
