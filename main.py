#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
                                                        #logístico multinomial
                                                        
import warnings
warnings.filterwarnings('ignore')
                                                        
# In[1]: Importing data:
data = pd.read_csv('/Users/rodrigomendes/Documents/projects/titanic/train.csv')

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
data['phat_bool'] = 1 - round(data['phat']) 
data['phat'] = 1 - data['phat']
data[['Survived','phat_bool' ,'phat']]

data['Survived'] = data['Survived'].astype('float')
assertivity = data[data['Survived'] == data['phat_bool']]
accuracy = len(assertivity)/len(data)
print('Acuracia: ' + str(accuracy))

# In[6]: Validando acuracia do modelo com dados externos:
    
data_test = pd.read_csv('/Users/rodrigomendes/Documents/projects/titanic/test.csv')
data_test.info()

# In[6.5]: 
data_test['phat'] = model.predict(data_test)
data_test['phat_bool'] = 1 - round(data_test['phat']) 
data_test['phat'] = 1 - data['phat']
data_test[['phat_bool' ,'phat']]

data_test['Survived'] = data_test['phat_bool'].astype('category')
data_test['Survived'] = data_test['Survived'].fillna(0)
data_test[['PassengerId', 'Survived']].to_csv('output.csv')

 
