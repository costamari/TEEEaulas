# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:02:30 2021

@author: gomes
"""

# Mariana Gomes Costa - mat: 119111325

# classificador QEE

import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import numpy as np

# pip install scikit-plot
import scikitplot as skplt
import matplotlib.pyplot as plt

# --------------------------- entradas
entradas = pd.read_csv('entradas_QEE.csv', sep=",", header=None)
alvos = pd.read_csv('alvos_QEE.csv', sep=",", header=None)


# --------------------------- treinamento e teste
Ent_tre, Ent_test, Alvo_tre, Alvo_test = train_test_split(entradas, alvos, 
                                                          test_size=0.3, shuffle=True)

#--------------------------- classificador
net = MLPClassifier(solver='lbfgs', max_iter = 500, alpha=1e-6, hidden_layer_sizes=77)


# --------------------------- modelo ajustado
modelo_ajustado = net.fit(Ent_tre, Alvo_tre)

# ---------------------------- estimação 
score = modelo_ajustado.score(Ent_test, Alvo_test)

# ----------------------------- previsão
previsoes = modelo_ajustado.predict(Ent_test)
prevpb = modelo_ajustado.predict_proba(Ent_test)

precisao = accuracy_score(Alvo_test, previsoes)
print(precisao)

print(classification_report(Alvo_test, previsoes))


# ------------------------------- matriz de confusão
confusao = confusion_matrix(Alvo_test.values.argmax(axis=1), previsoes.argmax(axis=1))
print(confusao)


opcoes_titulos = [("Matriz de confusão sem normalização", None),
                  ("Matriz de confusão normalizada", 'true')]

df_confusion = pd.crosstab(Alvo_test.values.argmax(axis=1), previsoes.argmax(axis=1),
                           rownames=['Actual'], colnames=['Predicted'], margins=False)


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation='horizontal')
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
plot_confusion_matrix(df_confusion)
plt.show()

"""
# Plota utilizando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix(Alvo_test, previsoes)
plt.show()

# Plotar a ROC
skplt.metrics.plot_roc(Alvo_test, prevpb)
plt.show()

skplt.metrics.plot_precision_recall(Alvo_test, prevpb)
plt.show()"""