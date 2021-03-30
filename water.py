
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
train_data = pd.read_csv("models/trans/train.csv")
train_data.head()

test_data = pd.read_csv("models/trans/test.csv")
test_data.head()

#%%
val_dataframe = train_data.sample(frac=0.2, random_state=1337)
train_dataframe = train_data.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)
#%%
features = ["Pclass", "Sex", "SibSp", "Parch","Survived"]
train_dataframe = pd.get_dummies(train_dataframe[features])
val_dataframe = pd.get_dummies(val_dataframe[features])

train_dataframe_Y = train_dataframe.pop("Survived")

#%%

#optional?
val_dataframe_X = val_dataframe[["Pclass","SibSp","Parch","Sex_female","Sex_male"]]
val_dataframe_Y = val_dataframe[["Survived"]]

#%%

X_test = pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])
#%%

# val_dataframe = val_dataframe.to_numpy()
train_dataframe_Y = train_dataframe_Y.to_numpy()
train_dataframe = train_dataframe.to_numpy()

#%%

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:\\Users\\CB\\PycharmProjects\\tryout\\models\\logs",histogram_freq=1,profile_batch = 100000000)
# python -m tensorboard.main --logdir $PWD/logs
# %load_ext tensorboard
# %tensorboard --logdir logs --bind_all
#
#%%

model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss =  keras.losses.BinaryCrossentropy(),
                      optimizer = keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))



# tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

model.fit(train_dataframe, train_dataframe_Y, epochs=1000, callbacks =[tensorboard_callback])

# model.fit(train_dataframe, train_dataframe_Y, epochs=1000,
#           validation_data = [val_dataframe])



#%%
y_pred = model.predict_classes(val_dataframe_X, batch_size=64, verbose=1)
X_pred = model.predict_classes(X_test, batch_size=64, verbose=1)
#%%

from sklearn.metrics import classification_report

# y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(val_dataframe_Y, y_pred))

#%%