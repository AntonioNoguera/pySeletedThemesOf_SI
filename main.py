import eel

#pip install eel

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

#Se arranca la lib eel
eel.init('web',allowed_extensions=['.js','.html','.css'])

#Importación de Modelo
df_Analysis = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df_Cleared = pd.read_csv("heart_failure_clinical_records_dataset_cleared.csv")

#Funcion de Calculo
@eel.expose
def graficas():
    #Código para el análisis de datos

    #Desplegando las gráficas
    iter = 0 ;
    for i in df_Analysis:
        #Serie de tiempo 
        plt.subplot(2,3,iter%6+1)
        plt.plot(df_Analysis[i],color='#5885BF')
        plt.title(i)

        if(iter%6 ==5 or iter == df_Analysis.shape[1]-1):  
            plt.tight_layout()
            plt.show()
        iter = iter + 1

@eel.expose
def distribucion():
    iter = 0 ;
    for i in df_Analysis:
        #Serie de tiempo 
        plt.subplot(2,3,iter%6+1)
        plt.hist(df_Analysis[i], color='#5885BF')
        plt.title(i)

        if(iter%6 ==5 or iter == df_Analysis.shape[1]-1):
            plt.tight_layout()
            plt.show()
        iter = iter + 1

@eel.expose
def corr():
    correlation_mat = df_Analysis.corr(method='pearson')

    plt.figure(figsize=(11, 7))
    sns.heatmap(correlation_mat, annot=True, fmt=".2f", xticklabels=False, yticklabels=True, annot_kws={'size': 12})

    plt.tight_layout()
    plt.show() 

@eel.expose
def entrenarRed():
    xDF = df_Cleared.iloc[:,0:6]
    yDF = df_Cleared.iloc[:,6]
    

    x = xDF.to_numpy()
    y = yDF.to_numpy()
    print(xDF.head())

    from sklearn.metrics import accuracy_score 

    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10000,verbose=True, random_state=42)    
    model.fit(x,y)
    y_pred = model.predict(x)

    from sklearn.metrics import mean_absolute_percentage_error 
    accuracy = accuracy_score(y, y_pred)
    print("Precisión del MLPClassifier: {:.2f}%".format(accuracy * 100))
    
    plt.plot(y)
    plt.plot(y_pred,color='#FF0000') 
    plt.show()
    

#Se despliega el programa  
#eel.start('index.html',fullscreen=True, mode='chrome',size=(2000,2000))

entrenarRed()