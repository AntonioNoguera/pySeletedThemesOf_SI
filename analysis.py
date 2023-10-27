#Código para el análisis de datos

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
import random


#Importación de Modelo
df_Analysis = pd.read_csv("heart_failure_clinical_records_dataset.csv")

#Desplegando las gráficas

iter = 0 ;
for i in df_Analysis:
    #Serie de tiempo 
    plt.subplot(2,3,iter%6+1)
    plt.plot(df_Analysis[i],color='#5885BF')
    plt.title(i)

    if(iter%6 ==5 or iter == df_Analysis.shape[1]-1): 
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.tight_layout() 
        plt.show()
    iter = iter + 1

#Desplegando las distribución
iter = 0 ;
for i in df_Analysis:
    #Serie de tiempo 
    plt.subplot(2,3,iter%6+1)
    plt.hist(df_Analysis[i], color='#5885BF')
    plt.title(i)

    if(iter%6 ==5 or iter == df_Analysis.shape[1]-1):
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.tight_layout() 
        plt.show()
    iter = iter + 1
