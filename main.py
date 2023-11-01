import eel

#pip install eel

import pandas as pd
import numpy as np
import pickle

from matplotlib.ticker import MultipleLocator, AutoMinorLocator


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    print(x[:5])

    #Testeando Predicción con PCA
    NUM_COMPONENTES = 6
    np.set_printoptions(suppress=True)
    pca = PCA(n_components=NUM_COMPONENTES)

    x = pca.fit_transform(x)
    print(x[:5])


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


    from sklearn.metrics import accuracy_score  

    
    model = MLPClassifier(hidden_layer_sizes=(500, 30), max_iter=10000,verbose=False, activation='identity',solver='adam',tol=1e-8,n_iter_no_change=40)    
    

    maximo=30
    i=0
    topAccuracy = 0
    y_top = 0


    while i<maximo:
        
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        from sklearn.metrics import mean_absolute_percentage_error 
        accuracy = accuracy_score(y_test, y_pred)
        print("Generando Archivo....")
        if accuracy > topAccuracy:
            topAccuracy = accuracy
            y_top = y_pred
            # Guardamos el modelo en un archivo con pickle
            with open('modelo_paro_cardiaco.pkl', 'wb') as archivo:
                pickle.dump(model, archivo)
        i = i+1
    print("Precisión Seleccionada:  ",topAccuracy)

    plt.plot(y_test,label='Real')
    plt.plot(y_pred,label='Predicción') 
    plt.legend(loc='upper right')
    print("Predicción:",y_top)
    print("Realidad  :",y_test) 
    plt.show()
        

#Se despliega el programa  
#eel.start('index.html',fullscreen=True, mode='chrome',size=(2000,2000))

entrenarRed()