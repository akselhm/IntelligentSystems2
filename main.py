import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

read_file = pd.read_csv (r'data\breast-cancer-wisconsin.data')
read_file.to_csv (r'data\breast-cancer-wisconsin.csv', index=None)

dataset = pd.read_csv(
    "./data/breast-cancer-wisconsin.csv",
    names=["Id", "Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
           "Single Epithelial Cell Size", "Bare_Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])

print(len(dataset))
#dataset.head()

dataset = dataset[dataset.Bare_Nuclei != "?"] #remove rows with "?"

print(len(dataset))

# make test and train set

array = dataset.values
for i in range(len(array)):
    if array[i,-1] == 4:
        array[i, -1] = 1
    else:
        array[i, -1] = 0
print(len(array))
print(array[0])

tf.random.set_seed(42)

train, test = train_test_split(array, test_size=0.2, random_state=42)

train = np.asarray(train).astype('float32')
test = np.asarray(test).astype('float32')

print(train[0])



#make X and Y arrays

X_train = tf.convert_to_tensor(train[:, 1:-1]) # skip id
Y_train = tf.convert_to_tensor(train[:, -1])


X_test = tf.convert_to_tensor(test[:, 1:-1]) #skip id
Y_test = tf.convert_to_tensor(test[:, -1])


print(X_train[0])
print(Y_train[:10])
# make class for neural network
"""

X = tf.convert_to_tensor(array[:, 1:-1]) # skip id
Y = tf.convert_to_tensor(array[:, -1])
"""

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(9, activation="relu", name="layer1"),
        layers.Dense(6, activation="relu", name="layer2"),
        layers.Dense(3, activation="relu", name="layer3"),
        layers.Dense(1, name="layer4"),
    ]
)

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]
)

history = model.fit(X_train, Y_train, epochs=20)

# Call model on a test input
#x = X_train[0]
#y = model(x)

#make predictions

predictions = model.predict(X_test)

#turn into classes
prediction_classes = [
    1 if prob > 0.5 else 0 for prob in np.ravel(predictions)
]

print(f'Accuracy: {accuracy_score(Y_test, prediction_classes):.2f}')
print(model.layers)




