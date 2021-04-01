#%%

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_core.python.keras import regularizers

#%%

train_data = pd.read_csv(r"C:\Users\CB\PycharmProjects\tryout\models\trans\train.csv")
train_data.head()

test_data = pd.read_csv(r"C:\Users\CB\PycharmProjects\tryout\models\trans\test.csv")
test_data.head()

#%%

features = ["Pclass", "Sex", "SibSp", "Parch","Survived"]
train_dataframe = pd.get_dummies(train_data[features])

train_dataframe_Y = train_dataframe.pop("Survived")
#%%


X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])
#%%

# val_dataframe = val_dataframe.to_numpy()
train_dataframe_Y = train_dataframe_Y.to_numpy()
train_dataframe = train_dataframe.to_numpy()
#%%

model = tf.keras.Sequential([
    layers.Dense(16,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.01),
               ),
  layers.Dense(units=1, activation='sigmoid')
])


model.compile(loss = tf.keras.losses.Hinge(reduction="auto", name="categorical_hinge"),
                      optimizer = tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True))

model.fit(train_dataframe, train_dataframe_Y, epochs=1000)

# model.fit(train_dataframe, train_dataframe_Y, epochs=1000,
#           validation_data = [val_dataframe])

#%%
# y_pred = model.predict_classes(val_dataframe_X, batch_size=64, verbose=1)
X_pred = model.predict_classes(X_test, batch_size=64, verbose=1)
train_dataframe_Y_pred = model.predict_classes(train_dataframe, batch_size=64, verbose=1)
#%%

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

print(classification_report(train_dataframe_Y, train_dataframe_Y_pred))

print(accuracy_score(train_dataframe_Y, train_dataframe_Y_pred))

print(confusion_matrix(train_dataframe_Y, train_dataframe_Y_pred))


#%%
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': X_pred.ravel()})
output.to_csv(r'C:\Users\CB\PycharmProjects\tryout\my_sub.csv', index=False)
print("Your submission was successfully saved!")

#
#%%

model = tf.keras.Sequential([
  layers.Dense(16,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.01),
               ),

  layers.Dense(16,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.01),
               ),
    layers.Dense(16,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.01),
               ),
    layers.Dense(16,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.01),
               ),
  layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss =  tf.keras.losses.BinaryCrossentropy(),
                      optimizer = tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

model.fit(train_dataframe, train_dataframe_Y, epochs=1000,)

# model.fit(train_dataframe, train_dataframe_Y, epochs=1000,
#           validation_data = [val_dataframe])

