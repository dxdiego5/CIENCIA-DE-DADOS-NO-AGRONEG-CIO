import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from xgboost import plot_importance
import warnings
warnings.filterwarnings("ignore")

# conxao com dados
train_d = pd.read_csv('data/train.csv')

# print(train_d.head()) # view 10 first lines
# print(train_d.info()) # information the atributs


# explorando variavel de resposta
# train_d['Crop_Damage'].value_counts()
# ax = sns.countplot(x=train_d['Crop_Damage'])
# plt.show()

# explorando a variavel
# train_d['Season'].value_counts()
# ax = sns.countplot(x=train_d['Season'])
# plt.show()

# explorando a variavel
# train_d['Pesticide_Use_Category'].value_counts()
# ax = sns.countplot(x=train_d['Pesticide_Use_Category'])
# plt.show()

# explorando a variavel
# train_d['Crop_Type'].value_counts()
# ax = sns.countplot(x=train_d['Crop_Type'])
# plt.show()

# explorando a variavel
# train_d['Soil_Type'].value_counts()
# ax = sns.countplot(x=train_d['Soil_Type'])
# plt.show()

# Verificar valores nulos
# print(train_d.isnull().sum())

# Verificar valores duplicados
# print(train_d.duplicated().sum())

# Estatistica descritiva
# train_d['Estimated_Insects_Count'].describe()
# ax= sns.boxplot(x= train_d['Estimated_Insects_Count'])
# plt.show()

# Estatistica descritiva
# train_d['Number_Doses_Week'].describe()
# ax= sns.boxplot(x= train_d['Number_Doses_Week'])
# plt.show()

# Estatistica descritiva
# train_d['Number_Weeks_Used'].describe()
# ax= sns.boxplot(x= train_d['Number_Weeks_Used'])
# plt.show()

# Estatistica descritiva
# train_d['Number_Weeks_Quit'].describe()
# ax= sns.boxplot(x= train_d['Number_Weeks_Quit'])
# plt.show()


""""
Pre-processing dataset
Pre-processing dataset
Pre-processing dataset
"""
# excluir a variavel ID pois ela nao e explicativa
train_d.drop('ID', axis=1, inplace=True)

# exclusao de valores ausentes
train_d.dropna(inplace=True)

# One_Hot Enconder - Criaca ode variaveis numericas
for col in ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season']:
    train_d = pd.get_dummies(train_d, columns=[col])

# print(train_d.head(2))
# print(train_d.shape)

# Split dataset
x = train_d.drop(['Crop_Damage'], axis=1)
y = train_d['Crop_Damage'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=7)


""""
Construcao das maquinas preditivas
Construcao das maquinas preditivas
Construcao das maquinas preditivas
"""
Maquina_Preditiva = CatBoostClassifier(n_estimators=1000, max_depth=4,random_state=7)
Maquina_Preditiva.fit(X_train,Y_train)
predicoes = Maquina_Preditiva.predict(X_test)


""""
Avaliacoes das maquinas preditivas
Avaliacoes das maquinas preditivas
Avaliacoes das maquinas preditivas
"""

# Score do modulo nos dados de teste
result = Maquina_Preditiva.score(X_test,Y_test)
print("Acurácia nos Dados de Teste: %.3f%%" % (result * 100.0))

# Determinaas variaveis mais importantes
print(Maquina_Preditiva.feature_importances_)
print(train_d.info())