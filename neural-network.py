"""Using neural_network in python library sklearn and training on inbuilt breast cancer dataset"""

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#to see keys
cancer.keys()
#cancer object is like a dictionary, with keys as cancer.keys()
#to see values:
# cancer['data'] or cancer['target']
cancer['data'].shape

X=cancer['data']
y=cancer['target']

#Split data to train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y);

#Preprocessing
#Normalize data: Xi=(Xi-mean)/std
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler();
#Fit only to the training datasets
scaler.fit(X_train)
#Now transform the datasets
X_train = scaler.transform(X_train);
X_test  = scaler.transform(X_test);

#Train the model
from sklearn.neural_network import MLPClassifier
#Using 3 hidden layers of sizes 30 each
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30));
mlp.fit(X_train,y_train);
#By default the MLPClassifier is using 'relu' ativation function.
#We can change its default values

#We are done with training

#Testing
predictions=mlp.predict(X_test);
#Now we use confusion_matrices and generate classification report
from sklearnself.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions));
print(classification_report(y_test, predictions));


#We can also get the weights used by hidden layers and intercepts(theta_0 values)
#coefs=cofficients(theta_1...theta_n), intercepts=theta_0;
len(mlp.coefs_)
len(mlp.coefs_[0])
len(mlp.coefs_[0])

len(mlp.intercepts_)
len(mlp.intercepts_[0])
