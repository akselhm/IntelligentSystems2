import tensorflow as tf
import numpy as np
import pandas as pd
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
# make class for neural network
"""

X = tf.convert_to_tensor(array[:, 1:-1]) # skip id
Y = tf.convert_to_tensor(array[:, -1])
"""

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(10, activation="relu", name="layer1"),
        layers.Dense(5, activation="relu", name="layer2"),
        layers.Dense(1, name="layer3"),
    ]
)

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]
)

history = model.fit(X_train, Y_train, epochs=10)

# Call model on a test input
#x = X_train[0]
#y = model(x)

print(model.layers)




