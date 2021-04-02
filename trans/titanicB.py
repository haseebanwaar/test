

#TODO, introduce validation data in tensorboard.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import GridSearchCV
from pandas_profiling import ProfileReport

tf.random.set_seed = 100
np.random.seed(100)

# %%
train_data = pd.read_csv("models/trans/train.csv")
train_data.head()

test_data = pd.read_csv("models/trans/test.csv")
test_data.head()

#%%

profile = ProfileReport(df_train, title=f"Pandas Profiling Report for Titanic Dataset"
                        ,explorative=True
                        #,samples=None
                        #,correlations=None
                        #,missing_diagrams=None
                        #,duplicates=None
                        #,interactions=None
                       )
profile.to_file("profile.html")
display(profile)

#%%
val_dataframe = train_data.sample(frac=0.2, random_state=1337)
train_dataframe = train_data.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)
#%%
features = ["Fare","Age","Pclass", "Sex", "SibSp", "Parch","Survived"]
train_dataframe = pd.get_dummies(train_dataframe[features])
val_dataframe = pd.get_dummies(val_dataframe[features])
train_dataframe= train_dataframe.dropna()
val_dataframe= val_dataframe.dropna()


#%%
train_dataframe_Y = train_dataframe.pop("Survived")

val_dataframe_Y = val_dataframe.pop("Survived")
#%%
scaler = preprocessing.StandardScaler().fit(train_dataframe)
train_dataframe = scaler.transform(train_dataframe)
# scaler = preprocessing.StandardScaler().fit(train_dataframe)
val_dataframe = scaler.transform(val_dataframe)

#%%

#optional?

#%%

X_test = pd.get_dummies(test_data[["Fare","Age","Pclass", "Sex", "SibSp", "Parch"]])
#%%
X_test = scaler.transform(X_test)
#%%

# val_dataframe = val_dataframe.to_numpy()
train_dataframe_Y = train_dataframe_Y.to_numpy()
# train_dataframe = train_dataframe.to_numpy()


#%%
i=1
#%%
from datetime import datetime
# log_dir = './models/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
# i = 1
i+=1

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fr".\models\logs\{i}",histogram_freq=1,
                                                      profile_batch = 0,update_freq='epoch')

# python -m tensorboard.main --logdir C:\\Users\\CB\\PycharmProjects\\tryout\\models\\logs
# %load_ext tensorboard
# %tensorboard --logdir logs --bind_all
#
#%%
make an ensemle
submit fee
scale only the contuous variables
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
df = pd.concat([traindf, testdf], axis=0, sort=False)





#%%




i+=1

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fr".\models\logs\{i}",histogram_freq=1,
                                                      profile_batch = 0,update_freq='epoch')

model = tf.keras.Sequential([
    layers.Dense(32,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.001),
               ),
    layers.Dropout(0.2),
    layers.Dense(32,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.001),
               ),
    layers.Dense(32,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.001),
               ),
  layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss =  tf.keras.losses.Hinge(reduction="auto", name="categorical_hinge"),
# model.compile(loss =  tf.keras.losses.BinaryCrossentropy(),
                      optimizer = tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True))

model.fit(train_dataframe, train_dataframe_Y, epochs=300,
          validation_split=0.2,callbacks =[tensorboard_callback])
print(model.summary())
# model.fit(train_dataframe, train_dataframe_Y, epochs=1000,
#           validation_data = [val_dataframe])
#%%
y_pred = model.predict_classes(val_dataframe, batch_size=64, verbose=1)
X_pred = model.predict_classes(X_test, batch_size=64, verbose=1)
train_dataframe_Y_pred = model.predict_classes(train_dataframe, batch_size=64, verbose=1)


#%%

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(val_dataframe_Y, y_pred))
print(classification_report(train_dataframe_Y, train_dataframe_Y_pred))
print(accuracy_score(val_dataframe_Y, y_pred))
print(accuracy_score(train_dataframe_Y, train_dataframe_Y_pred))

print(confusion_matrix(val_dataframe_Y, y_pred))
print(confusion_matrix(train_dataframe_Y, train_dataframe_Y_pred))

#%%
keras.utils.plot_model(model, show_shapes=True, rankdir="LR",to_file = './models/keras.png')


#%%
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': X_pred.ravel()})
output.to_csv('sub.csv', index=False)
print("Your submission was successfully saved!")

#
#%%

model = tf.keras.Sequential([
  layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),

  layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),

  layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),

  layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),

  layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),
    layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),
    layers.Dense(92,tf.keras.activations.relu,
               kernel_regularizer=regularizers.l2(0.0005),
               ),
  layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss =  tf.keras.losses.BinaryCrossentropy(),
                      optimizer = tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

model.fit(train_dataframe, train_dataframe_Y, epochs=1000, validation_data=[val_dataframe, val_dataframe_Y],
          callbacks =[tensorboard_callback])

# model.fit(train_dataframe, train_dataframe_Y, epochs=1000,
#           validation_data = [val_dataframe])

