# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:01:50 2021

@author: gomes
"""

# Mariana Gomes Costa - mat: 119111325

# classificador de para raios

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import scikitplot as skplt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# --------------------------- amostras
amostras_bom = pd.read_csv('corrente_PR_BOM.csv', header=None)
amostras_def = pd.read_csv('corrente_PR_DEF.csv', header=None)


index = np.random.randint(0, amostras_bom.values.shape[0])
plt.plot(amostras_bom.values[index])

# -------------------------------- pré processamento:
size_bom = amostras_bom.values.shape[1]
size_def = amostras_def.values.shape[1]

# -------------------------- reduz para 250 amostras
x_bom = amostras_bom.values[::, ::(size_bom//250)]
x_bom = x_bom[::, 50::] # Descarta as 50 primeiras (200 amostras restantes)

# -------------------------- reduz para 250 amostras
x_def = amostras_def.values[::, ::(size_def//250)]
x_def = x_def[::, 50::] # Descarta as 50 primeiras (200 amostras restantes)

# ------------------------- amostras depois do pré processamento:
plt.plot(x_bom[index])

# TREINAMENTO E TESTE

# ------------------------- Criando os alvos
y_bom = np.ones(x_bom.shape[0]) # SEM defeito
y_def = np.zeros(x_def.shape[0]) # COM defeito

x = np.concatenate((x_bom, x_def), axis=0)
y = np.concatenate((y_bom, y_def), axis=0)

#
# -------------------------------- Separando os bancos de teste e treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)


# ----------------------------------------- CLASSIFICADOR
net = MLPClassifier(solver = 'lbfgs', max_iter = 500, hidden_layer_sizes=(100), alpha=1e-6)

# treinando o classificador
modelo_ajustado = net.fit(x_train, y_train)

# AVALIANDO O MODELO
# Estima a precisao do modelo a partir da base de teste
score = modelo_ajustado.score(x_test, y_test)
print('Precisão:', score*100, '%')

# Calcula as previsoes do modelo a partir da base de teste
previsoes = modelo_ajustado.predict(x_test)
prevpb = modelo_ajustado.predict_proba(x_test)

precisao = accuracy_score(y_test, previsoes)
print('Acurácia:', precisao*100, '%')

print(classification_report(y_test, previsoes))


# PREVISÕES
index = np.random.randint(0, 20)

y_exemplo = y_test[index]
previsao = previsoes[index]
print('Rótulo:', 'Sem' if y_exemplo==1 else 'Com', 'defeito');
print('Previsão:', 'Sem' if previsao==1 else 'Com', 'defeito');
print('\n')

# PLOTAGEM
plt.plot(x_test[index])

skplt.metrics.plot_roc(y_test, prevpb)
plt.show()

skplt.metrics.plot_precision_recall(y_test, prevpb)
plt.show()

# import struct;print(struct.calcsize("P") * 8)