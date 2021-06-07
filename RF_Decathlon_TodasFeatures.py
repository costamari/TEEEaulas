# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:07:50 2021

@author: gomes
"""
# Mariana Gomes Costa 119111325

import pandas as pd

# ************** chamando a base de dados **************

df = pd.read_csv('Decathlon.csv', names=['Athets', '100m', 'Long jump', 
                                         'Shot put', 'High jump', '400m', 
                                         '110m hurdle', 'Discus', 
                                         'Pole vault', 'Javeline', '1500m',
                                         'Rank', 'Points', 'Competition'])
features = ['100m', 'Long jump', 'Shot put', 'High jump', '400m', '110m hurdle', 'Discus', 'Pole vault', 'Javeline', '1500m', 'Rank', 'Points']

# ************** separando os atributos **************

x = df.loc[:, features].values

# ************** separando a variável objetivo e transformando em numérica **************

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Competition'])

# ************** separando o conjunto de treinamento do conjunto de testes **************

from sklearn.model_selection import train_test_split

total = 0
acuracias = []
total_precisao = 0
precisao = []
total_recall = 0
recall = []

for i in range(5):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

    # Criando uma Decision Tree
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(n_estimators=100)

    # Treinando o modelo
    clf.fit(x_train,y_train)

    # Prevendo com o modelo
    y_pred = clf.predict(x_test)

    # Calcular métricas de avaliação de performance
    from sklearn import metrics

    Confusao = metrics.confusion_matrix(y_test, y_pred)
    Confusao = pd.DataFrame(Confusao, columns = ['Positivo','Negativo'])
    Confusao.index=['Positivo','Negativo']
    aux = metrics.accuracy_score(y_test, y_pred)
    total += aux
    acuracias.append(aux)
    aux_precisao = metrics.precision_score(y_test, y_pred)
    total_precisao += aux_precisao
    precisao.append(aux_precisao)
    
    aux_recall = metrics.recall_score(y_test, y_pred)
    total_recall += aux_recall
    recall.append(aux_recall)

print("Acurácia média do mododelo: ", total/5*100, "%")
print("Precisão média do mododelo: ", total_precisao/5*100, "%")
print("Recall média do mododelo: ", total_recall/5*100, "%")