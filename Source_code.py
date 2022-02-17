import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# model untuk ketiga classifier
cSVM = svm.SVC(kernel='linear')
cNB = GaussianNB()
cNN = MLPClassifier()

df = pd.read_csv("heart.csv")
X = df.drop('target', axis=1)  
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=109)
# latih classifier
cSVM = cSVM.fit(X_train, y_train)
cNB = cNB.fit(X_train, y_train)
cNN = cNN.fit(X_train, y_train)

# prediksi data test
Y_SVM = cSVM.predict(X_test)
Y_NB = cNB.predict(X_test)
Y_NN = cNN.predict(X_test)
# print akurasi
print("Akurasi SVM : ", accuracy_score(y_test, Y_SVM))
print("Akurasi Naive Bayes : ", accuracy_score(y_test, Y_NB))
print("Akurasi Neural Network : ", accuracy_score(y_test, Y_NN))

from sklearn.metrics import roc_auc_score
ASVM=roc_auc_score(y_test, Y_SVM)
ANB=roc_auc_score(y_test, Y_NB)
ANN=roc_auc_score(y_test, Y_NN)
print('AUC SVM: %.3f' % ASVM)
print('AUC Naive Bayes: %.3f' % ANB)
print('AUC Neural Network: %.3f' % ANN)