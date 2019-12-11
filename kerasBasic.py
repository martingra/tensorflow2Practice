import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('DATA/fake_reg.csv')
df.head()

#sns.pairplot(df)
#plt.show()




# We need the data in numpy for sklearn
X = df[['feature1','feature2']].values # Features
y = df['price'].values # Label

# Now, we can plit the data into train and test using sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Scale the data (normalize)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# CREATE THE MODEL
model = Sequential()

# 3 layers with 4 neurons each
model.add(Dense(4,activation='relu')) 
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

# And a final output for prediction. It should produce something like the spected output (prices in this case)
model.add(Dense(1))

# Compile the model. Metaparameters are set here
# WHICH LOSS FUNCTION SHOULD I CHOOSE??
# For multi-class classification: categorical_crossentropy
# For binary classification: binary_crossentropy
# For regression: mse (mean squared error)
model.compile(optimizer='rmsprop',loss='mse')


# TRAIN THE MODEL

model.fit(x=X_train, y=Y_train, epochs=300)

# we can get information about the loss history and plot it using pandas.
loss_df = pd.DataFrame(model.history.history)
# loss_df.plot()


# TEST THE MODEL

model.evaluate(x=X_test, y=Y_test, verbose=0)

# we can get the predicted value for each input (X_test)
test_predictions = model.predict(X_test)

# we can now compare it with the real value that we have in Y_test
pred_df = pd.DataFrame(Y_test,columns=['Test Y'])
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test Y','Model Predictions']

sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)

pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']
sns.distplot(pred_df['Error'],bins=50)



mae = mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])
print(mae)
mse = mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])
print(mse)
mrse = mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])**0.5
print(mrse)


# we can save the model
model.save('my_model.h5')

# and we can load model 
mdl = load_model('my_model.h5')

# and use now a custom sample for example
sample = [[990,1100]]
# we need to scale (normalize) it
sample = scaler.transform(sample)
# and now we can predict
sample_output = mdl.predict(sample)
print(sample_output)


plt.show()