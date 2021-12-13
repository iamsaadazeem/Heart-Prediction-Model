import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix

heart_data = pd.read_csv("/Users/saadazeem/Downloads/heart_failure_clinical_records_dataset.csv")

#Training
Features = ['ejection_fraction','serum_creatinine','sex', 'platelets', 'smoking']
x = heart_data[Features]
y = heart_data["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1, random_state=50)

#Standard scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#random forest classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train, y_train)
pred_rfc = rfc.predict(x_test)

#How the model performed
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, pred_rfc)
ac = accuracy_score(y_test, pred_rfc)

#2-Way table (picure representation)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


#SVC classifier
clf = SVC()
clf.fit(x_train, y_train)
pred_clf = clf.predict(x_test)

#How the model performed
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))

#2-Way table (picure representation)
cm = confusion_matrix(y_test, pred_clf)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

#KNN
kn_clf = KNeighborsClassifier(n_neighbors=6)
kn_clf.fit(x_train, y_train)
pred_kn = kn_clf.predict(x_test)

#How the model performed
print(classification_report(y_test, pred_kn))
print(confusion_matrix(y_test, pred_kn))

cm = confusion_matrix(y_test, pred_kn)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("KNN Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


