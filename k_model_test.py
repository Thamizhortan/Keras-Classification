
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.pipeline import Pipeline
# Load data
data = pd.read_csv('Data/data.csv', delimiter=',')

# Head method show first 5 rows of data
print(data.head())

# Drop unused columns
columns = ['Unnamed: 32', 'id', 'diagnosis']

# Convert strings -> integers
d = {'M': 0, 'B': 1}

# Define features and labels
y = data['diagnosis'].map(d)
X = data.drop(columns, axis=1)

# Plot number of M - malignant and B - benign cancer

ax = sns.countplot(y, label="Count", palette="muted")
B, M = y.value_counts()
plt.savefig('Plots/count.png')
print('Number of benign cancer: ', B)
print('Number of malignant cancer: ', M)

# Split dataset into training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize data
X_train_N = (X_train-X_train.mean())/(X_train.max()-X_train.min())
X_test_N = (X_test-X_train.mean())/(X_test.max()-X_test.min())

####### PCA ######


# PCA without std
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('% of variance')
plt.title('PCA without Std')
plt.savefig('Plots/pcavariancewithoutstd.png')

# PCA with std
pca = PCA(n_components=6)
X_std = StandardScaler().fit_transform(X)
pca.fit(X_std)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('% of variance')
plt.title('PCA with Std')
#plt.savefig('Plots/pcavariancewithstd.png')

###### Neural Networks ######
svc = svm.SVC(kernel='linear', C=1)

# Pipeline
model = Pipeline([
    ('reduce_dim', pca),
    ('svc', svc)
])

# Fit
model.fit(X_train_N, y_train)
svm_score = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("SVM accuracy: %0.2f (+/- %0.2f)" % (svm_score.mean(), svm_score.std() * 2))



scaler = StandardScaler()

num_epoch = 10

# 1-layer NN
def l1neuralNetwork():
    model = Sequential()
    model.add(Dense(input_dim=30, units=2))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.summary()

    model.fit(scaler.fit_transform(X_train_N), y_train, epochs=num_epoch,
              shuffle=True)
    y_pred = model.predict_classes(scaler.transform(X_test_N.values))
    print("\n\naccuracy of 1-layer NN", np.sum(y_pred == y_test) / float(len(y_test)))


l1neuralNetwork()

# 3-layer NN
def l3neuralNetwork():
    model = Sequential()
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=2))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.summary()
    model.fit(scaler.fit_transform(X_train_N), y_train, epochs=num_epoch,
              shuffle=True)
    y_pred = model.predict_classes(scaler.transform(X_test_N.values))
    print("\n\naccuracy of 3-layer NN", np.sum(y_pred == y_test) / float(len(y_test)))

l3neuralNetwork()

# 5-layer NN
def l5neuralNetwork():
    model = Sequential()
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=30))
    model.add(Dense(input_dim=30, units=2))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.summary()
    model.fit(scaler.fit_transform(X_train_N), y_train, epochs=num_epoch,
              shuffle=True)
    y_pred = model.predict_classes(scaler.transform(X_test_N.values))
    print("\n\naccuracy of 5-layer NN", np.sum(y_pred == y_test) / float(len(y_test)))

l5neuralNetwork()


###### Classification Mark ######

# Confusion Matrix

y_pred = model.predict(X_test_N)
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)#for label size
cm_plot = sns.heatmap(df_cm, annot=True, fmt='n', annot_kws={"size": 12})# font size
cm_plot.figure.savefig('Plots/confusionmatrix.png')
#plt.show()

# Precision & Recall Score

print("Precision score {}%".format(round(precision_score(y_test, y_pred),3)))
print("Recall score {}%".format(round(recall_score(y_test, y_pred),3)))
print("F1 Score {}%".format(round(f1_score(y_test, y_pred, average='weighted'),3)))

# ROC Curve
y_score = model.fit(X_train_N, y_train).decision_function(X_test_N)

fpr, tpr, thresholds = roc_curve(y_test, y_score)


fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve for SVM')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curve for Breast Cancer Classifer')
plt.legend(loc="lower right")
plt.savefig('Plots/roccurve.png')

# Correlation Map

plt.figure()
f, ax = plt.subplots(figsize=(14,14))
corr_plot = sns.heatmap(X.corr(), annot=False, linewidths=.5, fmt='.1f', ax=ax)
corr_plot.figure.savefig('Plots/corrmap.png')
#plt.show()