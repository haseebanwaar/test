# TODO, introduce validation data in tensorboard.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import GridSearchCV
from pandas_profiling import ProfileReport
import re

# %%

tf.random.set_seed = 200
np.random.seed(100)

# %%
train_dataframe = pd.read_csv(r"C:\Users\CB\PycharmProjects\tryout\models\trans\train.csv")
train_dataframe.head()

test_data = pd.read_csv(r"C:\Users\CB\PycharmProjects\tryout\models\trans\test.csv")
test_data.head()

# %%

profile = ProfileReport(train_dataframe, title=f"Pandas Profiling Report for Titanic Dataset"
                        , explorative=True
                        # ,samples=None
                        # ,correlations=None
                        # ,missing_diagrams=None
                        # ,duplicates=None
                        # ,interactions=None
                        )
profile.to_file("profile.html")
# display(profile)

# %%
# val_dataframe = train_data.sample(frac=0.2, random_state=1337)
# train_dataframe = train_data.drop(val_dataframe.index)
#
# print(
#     "Using %d samples for training and %d for validation"
#     % (len(train_dataframe), len(val_dataframe))
# )
# %%
df = pd.concat([train_dataframe, test_data])
df['Title'] = df.Name.apply(lambda x: re.findall("[a-zA-Z]+\.", x)[0][:-1])

df['Title'] = df.Title.apply(
    lambda x: 'Mr' if x in ['Sir', 'Don', 'Jonkheer', 'Dr', 'Col', 'Capt', 'Major', 'Rev'] else x)
df['Title'] = df.Title.apply(lambda x: 'Mrs' if x in ['Lady', 'Countess', 'Dona', 'Ms', 'Mme', 'Mlle'] else x)
df['Title'].unique()
# %%
df.Embarked.fillna('S', inplace=True)
df.Fare.fillna(df.Fare.mean(), inplace=True)

df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'Age'] = 5
df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'Age'] = 22
df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'Age'] = 33
df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'Age'] = 37

# Selected features
features = ["Title", "Fare", "Age", "Pclass", "Sex", "SibSp", "Parch", "Embarked", "Survived"]

# Get dummies for categorical variables
df = pd.get_dummies(df[features])
# %%

test_X = df[df.Survived.isna()].drop('Survived', axis=1)
# Pop Label 'Survived' from train and place it into train_Y
train_X = df.dropna()
train_Y = train_X.pop("Survived")

# %%

# Scale continuous variabels for better model
scaler = preprocessing.StandardScaler().fit(df)
df = scaler.transform(df)

# %%

# val_dataframe = val_dataframe.to_numpy()
train_X = train_X.to_numpy()
# train_dataframe = train_dataframe.to_numpy()


# %%
i = 1
# %%
from datetime import datetime

# log_dir = './models/logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")

# python -m tensorboard.main --logdir C:\\Users\\CB\\PycharmProjects\\tryout\\models\\logs
# %load_ext tensorboard
# %tensorboard --logdir logs --bind_all

make
an
ensemle
submit
fee
scale
only
the
contuous
variables


#%%
def model_architecture(no_of_layers, no_of_neurons, regularize):
    model = tf.keras.Sequential()
    for l in range(no_of_layers):
        model.add(
            layers.Dense(no_of_neurons, tf.keras.activations.relu,
                         kernel_regularizer=regularizers.l2(regularize),
                         ),

        )

    model.add(layers.Dense(units=1, activation='sigmoid'))
    return model



# %%
i+=1
models_summary = pd.DataFrame({
    'learning_rate': pd.Series([], dtype='str'),
    'n_layers': pd.Series([], dtype='int'),
    'n_neurons': pd.Series([], dtype='int'),
    'regularization_rate': pd.Series([], dtype='float'),
    'loss': pd.Series([], dtype='str'),
    'epochs': pd.Series([], dtype='int'),
    'Final_Loss': pd.Series([], dtype='float'),
    'Final_Accuracy': pd.Series([], dtype='float'),
    'Final_Precision': pd.Series([], dtype='float'),
    'Final_Recall' : pd.Series([], dtype='float'),
    'Final_Avg_PR' : pd.Series([], dtype='float'),
    'Final_Precision': pd.Series([], dtype='float'),
    'Final_Recall' : pd.Series([], dtype='float'),
    'Final_Avg_PR' : pd.Series([], dtype='float'),

})



learning_rate = [0.005, 0.0005]
n_layers = [1, 2, 3, 4]
n_neurons = [8, 32, 64]
regularization_rate = [0.001, 0.0001]
loss = ["binary_crossentropy"]
epochs = [150, 300]


# lrc = tf.keras.optimizers.schedules.ExponentialDecay(0.005,decay_steps=100,decay_rate=0.96,)
# lrcc = tf.keras.callbacks.LearningRateScheduler(lrc,verbose=1)
# lrc = tf.keras.optimizers.schedules.ExponentialDecay(0.005,decay_steps=100,decay_rate=0.96,)
# lrcc = tf.keras.callbacks.ReduceLROnPlateau()

for a in learning_rate:
    for b in n_layers:
        for c in n_neurons:
            for d in regularization_rate:
                for e in loss:
                    for f in epochs:
                        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            # log_dir=fr".\models\logs\{a},{b},{c},{d},{e},{f},",
                            # log_dir=fr".\models\logs\{i}",
                            # profile_batch=0, update_freq='epoch')
                        model = model_architecture(b, c, d)

                        if a != 'dynamic':
                            model.compile(loss=e, optimizer=tf.keras.optimizers.Adam(a), metrics=['accuracy',
                                                                                                 keras.metrics.Precision(),
                                                                                                 keras.metrics.Recall(),
                                                                                                 keras.metrics.TruePositives(),
                                                                                                 keras.metrics.TrueNegatives(),
                                                                                                  ])
                            model.fit(train_X, train_Y, epochs=f,
                                  # validation_split=0.2, callbacks=[tensorboard_callback])
                                  validation_split=0.2)

                        models_summary = models_summary.append({
                            'learning_rate': a,
                            'n_layers': b,
                            'n_neurons': c,
                            'regularization_rate': d,
                            'loss': e,
                            'epochs': f,
                            'Final_Loss': model.metrics[0].result().numpy(),
                            'Final_Accuracy': model.metrics[1].result().numpy(),
                            'Final_Precision': model.metrics[2].result().numpy(),
                            'Final_Recall': model.metrics[3].result().numpy(),
                            'Final_Avg_PR': (model.metrics[2].result().numpy()+model.metrics[3].result().numpy())/2,
                            'Final_TP': model.metrics[4].result().numpy(),
                            'Final_TN': model.metrics[5].result().numpy(),
                            'Final_Avg_PN': (model.metrics[4].result().numpy()+model.metrics[5].result().numpy())/2,
                        }, ignore_index=True)
                        model.save(f'{a}{b}{c}{d}{e}{f}')

# %%
i+=1
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fr".\models\logs\{i}",
                                                      histogram_freq=1,
                                                      profile_batch=0, update_freq='epoch')
model = model_architecture(4,32,0.0005)

model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy',
                                                                      keras.metrics.Precision(),
                                                                      keras.metrics.FalseNegatives(),
                                                                      keras.metrics.TruePositives(),
                                                                      keras.metrics.TrueNegatives(),
                                                                      keras.metrics.FalsePositives(),
                                                                      keras.metrics.Recall()])
model.fit(train_X, train_Y, epochs=700,
          validation_split=0.3, callbacks=[tensorboard_callback])



# %%
train_pred = model.predict_classes(train_X, verbose=1)
test_pred = model.predict_classes(test_X, verbose=1)
# train_dataframe_Y_pred = model.predict_classes(train_dataframe, batch_size=64, verbose=1)

# %%

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# print(classification_report(val_dataframe_Y, y_pred))
print(classification_report(train_Y, train_pred))
# print(accuracy_score(val_dataframe_Y, y_pred))
print(accuracy_score(train_Y, train_pred))

# print(confusion_matrix(val_dataframe_Y, y_pred))
print(confusion_matrix(train_Y, train_pred))

# %%
keras.utils.plot_model(model, show_shapes=True, rankdir="LR", to_file='./models/keras.png')

# %%
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_pred.ravel()})
output.to_csv('aaaaa.csv', index=False)
print("Your submission was successfully saved!")


#%%
p1 = pd.read_csv('sub.csv')
p2 = pd.read_csv('sub1.csv')
p3 = pd.read_csv('sub2.csv')
df1 = p1
#%%
df1['s1'] = p1.Survived
df1['s3'] = p3.Survived
#%%
model = keras.models.load_model('0.005180.0001binary_crossentropy300')
models_summary['aa'] = models_summary['Final_Avg_PR']+models_summary['Final_Avg_PN']