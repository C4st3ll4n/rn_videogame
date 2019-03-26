import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv('games.csv')
base = base.dropna(axis=0)

base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis = 1)
base = base.drop('Name', axis = 1)

base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)

base = base.loc[base['Global_Sales'] > 0.01]

previsores = base.iloc[:,[0,1,2,3,5,6,7,8]].values
vendaGlobal = base.iloc[:,4]

lb = LabelEncoder()
previsores[:, 0] = lb.fit_transform(previsores[:, 0])
previsores[:, 2] = lb.fit_transform(previsores[:, 2])
previsores[:, 3] = lb.fit_transform(previsores[:, 3])
previsores[:, 5] = lb.fit_transform(previsores[:, 5])

ohe = OneHotEncoder(n_values='auto',categorical_features=[0,2,3,5])
previsores = ohe.fit_transform(previsores).toarray()

ce = Input(shape=(396,))

co1 = Dense(units=186, activation='sigmoid')(ce)
co2 = Dense(units=186, activation='sigmoid')(co1)

cs1 = Dense(units=1, activation='linear')(co2)

regressor = Model(inputs = ce, 
                  outputs=cs1)

regressor.compile(optimizer='adam', loss='mae')
regressor.fit(previsores, [vendaGlobal], epochs=1000, batch_size=1000  )

previsaoGlobal = regressor.predict(previsores)




