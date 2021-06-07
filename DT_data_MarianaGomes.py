# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:56:05 2021

@author: gomes
"""

# Mariana Gomes Costa 119111325

import pandas as pd

# ************** chamando a base de dados **************

df = pd.read_csv('data.csv')

del df['Unnamed: 32'] # exclusão 

features = ['radius_mean', 'texture_mean','perimeter_mean', 'area_mean', 'smoothness_mean',
                                    'compactness_mean', 'concavity_mean', 'concave points_mean',
                                    'symmetry_mean', 'fractal_dimension_mean',
                                    'radius_se', 'texture_se', 'perimeter_se',
                                    'area_se', 'smoothness_se', 'compactness_se',
                                    'concavity_se', 'concave points_se', 'symmetry_se',
                                    'fractal_dimension_se', 'radius_worst', 'texture_worst',
                                    'perimeter_worst', 'area_worst', 'smoothness_worst',
                                    'compactness_worst', 'concavity_worst', 'concave points_worst',
                                    'symmetry_worst', 'fractal_dimension_worst']


# ************** separando os atributos **************

x = df.loc[:, features].values

# ************** separando a variável objetivo e transformando em numérica **************

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['diagnosis'])

# ************** separando o conjunto de treinamento do conjunto de testes **************

from sklearn.model_selection import  train_test_split

total_precisao = 0
total_acuracia = 0
total_recall = 0
acuracias = []
precisao = []
recall = []

for i in range(5):

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

    # Criando uma Decision Tree
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()

    # Treinando o modelo
    clf.fit(x_treino,y_treino)

    # Prevendo com o modelo
    y_pred = clf.predict(x_teste)

    # Calcular métricas de avaliação de performance
    from sklearn import metrics

    Confusao = metrics.confusion_matrix(y_teste, y_pred)
    aux_acuracia = metrics.accuracy_score(y_teste, y_pred)
    total_acuracia += aux_acuracia
    acuracias.append(aux_acuracia)
    
    aux_precisao = metrics.precision_score(y_teste, y_pred)
    total_precisao += aux_precisao
    precisao.append(aux_precisao)
    
    aux_recall = metrics.recall_score(y_teste, y_pred)
    total_recall += aux_recall
    recall.append(aux_recall)

print("Acurácia média do mododelo: ", total_acuracia/5*100, "%")
print("Precisão média do mododelo: ", total_precisao/5*100, "%")
print("Recall média do mododelo: ", total_recall/5*100, "%")