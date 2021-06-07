#Atividade 1 : Decathlon
#Mariana Gomes Costa

# ***************************************************************

# Análise exploratória de dados: bibliotecas utilizadas 

import pandas as pd
import numpy as np
from statistics import variance
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ************** chamando a base de dados **************

df = pd.read_csv('Decathlon.csv', names=['Athets', '100m', 'Long jump', 
                                         'Shot put', 'High jump', '400m', 
                                         '110m hurdle', 'Discus', 
                                         'Pole vault', 'Javeline', '1500m',
                                         'Rank', 'Points', 'Competition'])
features = ['100m', 'Long jump', 'Shot put', 'High jump', '400m', '110m hurdle', 'Discus', 'Pole vault', 'Javeline', '1500m', 'Rank', 'Points']

# ************** separando os atributos **************

x = df.loc[:, features].values

# ************** separando as classes (variável objetivo) **************

y = df.loc[:, ['Competition']].values

# ************** QUESTÃO 1) PARA VARIÂNCIA  **************

# nomes e competicao não entram pois são do tipo str

m_100 = x[:,0]
lj = x[:,1]
sp = x[:,2]
hj = x[:,3]
m_400 = x[:,4]
hurdle_100m = x[:,5]
discus = x[:,6]
pv = x[:,7]
jv = x[:,8]
m_1500 = x[:,9]
rank = x[:,10]
points = x[:,11]


# resultado das variâncias:
variancia = np.array ([variance(m_100), variance(lj), variance(sp), variance(hj),
                       variance(m_400), variance(hurdle_100m), variance(discus),
                       variance(pv), variance(jv), variance(m_1500), variance(rank),
                       variance(points)])

# visualizando os dados
    
fig = plt.figure(figsize =(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Points', fontsize = 15)
ax.set_ylabel('1500m', fontsize = 15)
ax.set_title('Representação em duas dimensões', fontsize = 20)
Competitions = ['Decastar', 'OlympicG']
colors = ['r', 'g']
for Competition, color in zip(Competitions, colors):
    indicesToKeep = df['Competition'] == Competition
    ax.scatter(df.loc[indicesToKeep, 'Points']
                , df.loc[indicesToKeep, '1500m']
                , c = color
                , s = 50)
    
ax.legend(Competitions)
ax.grid()
plt.show()

# transformando para csv
varianciacsv = pd.DataFrame(variancia)
varianciacsv.to_csv ('variancia.csv', index = True, header=False)

# ************** QUESTÃO 2) PARA CORRELAÇÃO  **************

label_encoder = LabelEncoder()
Competition = label_encoder.fit_transform(df['Competition'])
Competition = pd.DataFrame(Competition, columns = ['Competition'])

base = df.iloc[:, 0:13]
base = pd.concat([base, Competition], axis = 1)

correlacao = base.corr()

# visualizando os dados obtidos:

fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Rank', fontsize = 15)
ax.set_ylabel('Shot put', fontsize = 15)
ax.set_title('Representação em duas dimensões', fontsize = 20)
Competitions = ['Decastar', 'OlympicG']
colors = ['r', 'g']
for Competition, color in zip(Competitions, colors):
    indicesToKeep = df['Competition'] == Competition
    ax.scatter(df.loc[indicesToKeep, 'Rank']
               , df.loc[indicesToKeep, 'Shot put']
               , c = color
               , s = 50)
ax.legend(Competitions)
ax.grid()
plt.show()

# transformando para csv

correlacaocsv = pd.DataFrame(correlacao)
correlacaocsv.to_csv ('correlacao.csv', index = True, header=False)

# ************** QUESTÃO 3) PARA PCA 1  **************


# normalizando os atributos:
x = StandardScaler().fit_transform(x)

# calcular o PCA
pca = PCA(n_components = 12)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
                           , columns = ['Componente principal 1', 'Componente principal 2',
                                        'Componente principal 3', 'Componente principal 4',
                                        'Componente principal 5', 'Componente principal 6',
                                        'Componente principal 7', 'Componente principal 8',
                                        'Componente principal 9', 'Componente principal 10',
                                        'Componente principal 11', 'Componente principal 12'])

finalDf = pd.concat([principalDf, df[['Competition']]], axis = 1)

#transformando para csv

pcacsv = pd.DataFrame(finalDf)
pcacsv.to_csv ('finalDf.csv', index = True, header=False)

# Verificando os pesos que dormam as componentes principais:
pca.components_
pca.explained_variance_ratio_


# ************** QUESTÃO 3) PARA PCA 2  **************

# visualizar os dados em duas dimensões:
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Componente principal 7', fontsize = 15)
ax.set_ylabel('Componente principal 3', fontsize = 15)
ax.set_title('2 Componentes PCA', fontsize = 20)
Competitions = ['Decastar', 'OlympicG']
colors = ['r', 'g']
for Competition, color in zip(Competitions, colors):
    indicesToKeep = df['Competition'] == Competition
    ax.scatter(finalDf.loc[indicesToKeep, 'Componente principal 7']
               , finalDf.loc[indicesToKeep, 'Componente principal 3']
               , c = color
               , s = 50)
ax.legend(Competitions)
ax.grid()
plt.show()

# ************** QUESTÃO 3) PARA PCA 3  **************

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

def myplot(score, coeff, labels = None):
    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, "var" + str(i + 1), color = 'g')
        else:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, labels[i], color = 'g')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Componente principal 1 (%.2f)' %pca.explained_variance_ratio_[0])
    plt.ylabel('Componente principal 2 (%.2f)' %pca.explained_variance_ratio_[1])
    
    plt.grid()

myplot(principalComponents, np.transpose(pca.components_))
plt.show()