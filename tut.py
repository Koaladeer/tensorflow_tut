import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("diabetes.csv")
df.head()

for i in range(len(df.columns[:-1])):
    label = df.columns[i]
    plt.hist(df[df["Outcome"]==1][label],color='blue', label="Diabetes", alpha=0.6, density= True, bins=15)
    plt.hist(df[df["Outcome"]==0][label],color='red', label="No Diabetes", alpha=0.6, density= True, bins=15)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

x = df[df.columns[:-1]].values
y = df[df.columns[1]].values

scaler = StandardScaler()
x = scaler.fit_transform(x)
data = np.hstack((x,np.reshape(y,(-1,1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

over = RandomOverSampler()
x, y = over.fit_resample(x, y)
data = np.hstack((x, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns=df.columns)

len(transformed_df[transformed_df["Outcome"]==1]), len(transformed_df[transformed_df["Outcome"]==0])

x_train,x_temp, y_train, y_temp = train_test_split(x,y, test_size=0.4, random_state=0)
x_valid,x_test, y_valid, y_test = train_test_split(x_temp,y_temp, test_size=0.5, random_state=0)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics =['accuracy'])
model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_valid, y_valid))
model.evaluate(x_valid,y_valid)
