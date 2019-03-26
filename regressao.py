import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv('D:/Projetos/PycharmProjects/deeplearning/videogames/games.csv');
base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

base = base.dropna(axis = 0)

base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:,[0,1,2,3,7,8,9,10,11]].values
vendaNA = base.iloc[:,4].values
vendaEU = base.iloc[:,5].values
vendaJP = base.iloc[:,6].values

lb = LabelEncoder()
previsores[:, 0] = lb.fit_transform(previsores[:, 0])
previsores[:, 2] = lb.fit_transform(previsores[:, 2])
previsores[:, 3] = lb.fit_transform(previsores[:, 3])
previsores[:, 8] = lb.fit_transform(previsores[:, 8])

oneHot = OneHotEncoder(n_values='auto',categorical_features=[0,2,3,8])
previsores = oneHot.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,))

camada_oculta_1 = Dense(units=32, activation='sigmoid')(camada_entrada)
camada_oculta_2 = Dense(units=32, activation='sigmoid')(camada_oculta_1)

camada_saida_1 = Dense(units=1, activation='linear')(camada_oculta_2)
camada_saida_2 = Dense(units=1, activation='linear')(camada_oculta_2)
camada_saida_3 = Dense(units=1, activation='linear')(camada_oculta_2)

regressor = Model(inputs = camada_entrada, 
                  outputs=[camada_saida_1, camada_saida_2, camada_saida_3])

regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, [vendaNA, vendaEU, vendaJP], epochs=5000, batch_size=100  )

previsaoNA, previsaoEU, previsaoJP = regressor.predict(previsores)


























