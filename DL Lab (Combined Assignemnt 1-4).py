import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from CustomDL import *
from tensorflow.keras.datasets import mnist
t = 1.0
f = 0.0

#OR (Single Perceptron)
"""
X = np.array([[f,f],[f,t],[t,f],[t,t]])
Y = np.array([[f],[t],[t],[t]])

class perceptron:
    def __init__(self, rate = 0.1, n = 1000):
        self.lr = rate
        self.epochs = n
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for e in range(self.epochs):
            for i in range(X.shape[0]):
                Ypred = self.step(np.dot(self.weights,X[i])+self.bias)
                mae = Y[i] - Ypred
                self.weights += (self.lr * mae * X[i])
                self.bias += (self.lr * mae)

    def step(self,ac):
        if ac >= 0:
            return 1
        else:
            return 0
    def predict(self, Xtest):
        p = []
        for x in Xtest:
            p.append(self.step(np.dot(self.weights, x) + self.bias))
        return np.array(p)

p = perceptron()
p.fit(X,Y)

Xtest = np.array([[f,f],[t,f],[f,t],[f,f]])
Ytest = np.array([[f],[t],[t],[f]])

Ypred = p.predict(Xtest)
print(Ypred)
print("Accuracy - ", accuracy_score(Ytest,Ypred))

X = np.array([[f,f],[f,t],[t,f],[t,t]])
Y = np.array([[f],[t],[t],[t]])
"""

"""
#XOR (Multi Layer Perceptron MLP)
X = np.array([[f,f],[f,t],[t,f],[t,t]])
Y = np.array([[f],[t],[t],[f]])

Xtest = np.array([[t,f],[t,t],[f,t],[f,f]])
Ytest = np.array([[t],[f],[t],[f]])

model = Sequential()

model.add(Dense(4,input_shape=(2, ), activation = 'relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X,Y,epochs = 300)
loss,accuracy = model.evaluate(Xtest,Ytest,verbose = 0)
print("Accuracy- %.2f"%(accuracy*100))
print(loss)

"""
"""
#Glass
df = pd.read_csv("glass.csv")
df['Window'] = df["Type"].map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})
X  = df.drop(['Window', 'Type'], axis = 1).values
Y = df['Window'].values

class perceptron:
    def __init__(self, rate=0.1, n=1000):
        self.lr = rate
        self.epochs = n
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.E = []
        self.w_list = []

    def sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def feed(self, x):
        z = np.dot(x,self.w) + self.b
        h = self.sigmoid(z)
        return(h)

    def error(self,h, y):
        error = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        self.E = np.append(self.E, error)

    def backprop(self, X, y, h):
        self.delta_E_w = np.dot(X.T, h - y) / X.shape[0]
        self.delta_E_b = np.mean(h - y)

        self.w_list.append(self.w)

        self.w = self.w - self.lr * self.delta_E_w
        self.b = self.b - self.lr * self.delta_E_b

    def predict(self, X):
        pred = self.feed(X)
        return pred

    def classify(self, y):
        return self.predict(y).round()

    def train(self, X, y):
        for epoch in range(self.epochs):
            h = self.feed(X)

            self.backprop(X, y, h)

            self.error(h, y)


p = perceptron(rate = 0.02, n = 8000)
p.train(X, Y)

pred = p.classify(X)
acc = np.mean(pred == Y)
print("Acc - ", acc)
"""
"""
#MLR using MLP
data = pd.read_csv("MLRdata.csv")
x = data.drop("income", axis = 1).values
y = data["income"].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = Sequential()

model.add(Dense(4,input_shape=(xtrain.shape[1], ), activation = 'relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

model.fit(xtrain,ytrain,epochs = 300)
loss,mae = model.evaluate(xtest,ytest,verbose = 0)
print("MAE- ", mae)
print("RMSE- ", np.sqrt(loss))
"""
# MLP Backprop Abalone
"""
columns = [
    "Sex", "Length", "Diameter", "Height",
    "Whole_weight", "Shucked_weight",
    "Viscera_weight", "Shell_weight", "Rings"
]

data = pd.read_csv("abalone/abalone.data", names=columns)

data["Sex"] = data["Sex"].map({"M": 0, "F": 1, "I": 2})

x = data.drop("Rings", axis=1).values
y = data["Rings"].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = Sequential()
model.add(Dense(16, input_shape=(xtrain.shape[1],), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(xtrain, ytrain, epochs=100, batch_size=32, verbose=0)

mse, mae = model.evaluate(xtest, ytest, verbose=0)
rmse = np.sqrt(mse)
print("MAE model 1- ", mae)
print("RMSE model 1- ", rmse)


model2 = Automation()
model2.add(Layer(16, input_shape=(xtrain.shape[1],), activation="relu"))
model2.add(Layer(8, activation="relu"))
model2.add(Layer(1))

model2.run(optimizer="adam", loss="mse", metrics=["mae"])

model2.fit(xtrain, ytrain, epochs=100, verbose = 0)

mse, mae = model2.results(xtest, ytest)
rmse = np.sqrt(mse)

print("MAE model 2-", mae)
print("RMSE model 2-", rmse)

"""
#MNIST Digit Classification

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

y_train = one_hot(y_train)
y_test  = one_hot(y_test)

model = Automation()
model.add(Layer(128, input_shape=(784,), activation="relu"))
model.add(Layer(64, activation="relu"))
model.add(Layer(10, activation="softmax"))

model.run(loss="crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=64)

pred = model.predict(X_test)
acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))


print("Accuracy:", acc)
