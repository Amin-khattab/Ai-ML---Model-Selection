import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y = np.where(y == 2, 0, 1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

classifier = XGBClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

Fold = cross_val_score(estimator=classifier,X = x_train,y=y_train,cv=10)

print("the best(mean) accuracy is ",Fold.mean())
