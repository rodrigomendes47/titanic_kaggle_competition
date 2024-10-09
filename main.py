#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from statsmodels.iolib.summary2 import summary_col 
import numpy as np
import plotly.express as px                                                        
import warnings
warnings.filterwarnings('ignore')

# In[0.5]: Construção de função para a definição da matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)  
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores
                                                        
# In[1]: Importing data:
data = pd.read_csv('/Users/rodrigomendes/Documents/projects/titanic_kaggle_competition/train.csv')

data['Survived'] = data['Survived'].astype('category')
data['Pclass'] = data['Pclass'].astype('category')
data = data.dropna(subset=['Embarked', 'Age'])
data = data.drop(columns = ['Cabin', 'Embarked','Ticket'])

data.info()
data.describe()
# In[2]: Counting cases
data['Survived'].value_counts().sort_index()

# In[3]: Simple model:
data = data.drop(columns = ['PassengerId', 'Name'])

# In[3.5]: Simple model:
model = smf.glm(formula='Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare', data=data,
                         family=sm.families.Binomial()).fit()

# In[4]: Sumaring Model

model.summary()
summary_col([model],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

# In[5]: 
data['phat'] = model.predict()
data['phat'] = 1 - data['phat']
indicadores = matriz_confusao(data['phat'],data['Survived'].astype('float'),  0.60)
data['phat_bool'] = 1 - round(data['phat']) 
data[['Survived','phat_bool' ,'phat']]
print(indicadores)
# In[]
from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(data['Survived'], data['phat'])
roc_auc = auc(fpr, tpr)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) , fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()
# In[]


# In[6]: Validando acuracia do modelo com dados externos:
    
data_test = pd.read_csv('/Users/rodrigomendes/Documents/projects/titanic_kaggle_competition/test.csv')
data_test.info()

# In[6.5]: 
data_test['phat'] = model.predict(data_test)
data_test['phat_bool'] = 1 - round(data_test['phat']) 
data_test['phat'] = 1 - data['phat']
data_test[['phat_bool' ,'phat']]

data_test['Survived'] = data_test['phat_bool'].astype('category')
data_test['Survived'] = data_test['Survived'].fillna(0)
data_test[['PassengerId', 'Survived']].to_csv('output.csv')

 
