# -*- coding: utf-8 -*-
"""
Date : 05.03.2019
Spyder Editor
File : stats_model_churn_modeling.py
Carlos carvalho
Objectif : Data Preprocessing Stats Model

"""


import numpy as np
import pandas as pd
import pymysql
import sys
pd.options.display.max_columns = None
pd.options.display.max_rows = None

### Connect to the Database

def connMysql():
      # Parameters
      global conn
      Host="127.0.0.1",    # your host, usually localhost
      User="adm"
      Pwd="xxx",          
      Db="predicted_outputs",
      Socket="/Applications/MAMP/tmp/mysql/mysql.sock"
      conn = pymysql.connect(host="127.0.0.1", database="predicted_outputs", user = "adm", passwd= "xxx", unix_socket="/Applications/MAMP/tmp/mysql/mysql.sock")
      return conn
  

def getData():
    global data
    connMysql()
    sql = "SELECT * FROM predicted_outputs.churn_modelling_train"
    data = pd.read_sql(sql, conn)

getData()
 
### Résumer le Dataset

def df_preproc():
    ##if dataf:
    response = input("Nom du dataset a describe ? ")
    data = eval(response)
### Faire une description statistique du Dataset
    print ("#### Description statistique du Dataset")
    print(data.describe(include='all'))

### Lister les index de colonnes 
    cnt = 0
    print("### Indexes de colonnes")
    for p in data.columns:
        print (cnt,p)
        cnt=cnt+1
        
### Afficher les données manquantes
    print ("#### Afficher les données manquantes")
    print(data.isna().sum())
    
    
### Afficher les variables categoriques "type object"
    print ("#### Afficher les variables categoriques type object")
    print(data.dtypes)

### Afficher les details sur les variables du Datafarme
    print ("#### Nombre de lignes pour chaque variable")
    print(data.count())

    
df_preproc()

### Copier/Sauvegarder le DF initial vers le dataframe de PreProcessing
df2=copydf()

### Preparation
#Traiter les variables Categoriques - Un noyveau Dataframe est crée DF3

def treat_dum():
    global df3
    #df2=data_dum
    #df2=df_dummies
    df3=df2.copy()
    print ("Index number Colonne contenant les variables categoriques à traiter")
    #data_dum = input("Nom du dataset : ")
    response = ""
    while response != "n":
        idxCol = input(" Index de la colonne :")
        xCol=int(idxCol)
        print (df2.head(5),  xCol)
        dum_col=df3.iloc[:,[xCol]].columns
        df3=pd.get_dummies(df3, columns=dum_col)
        print(df3.head(5))
        print("### Indexes de colonnes")
        cnt = 0
        for p in df3.columns:
            print (cnt,p)
            cnt=cnt+1
        response = input(" Autres colonnes à traiter ?")
        continue

### Les variables selectionnes 4-Geo et 5-Gendre    
    
treat_dum()

### Choix des variables independantes du dataframe DF3

xCol1=[3,4,5,6,7,8,9,10,12,13,15]

#xCol1=[3,4,5,6,7,8,9,10,12,13]

#xCol1=[8,9,10,12,13]

def set_X_list(pdf,xCol1):
    #global X
    global X
    X=pdf.iloc[:,xCol1]
    #return X

set_X_list(df3,xCol1)

X.shape ### Matrcice Dim (1000,11)

### Choix de la vraible Independate y = 11 = Exited

def set_y(pdf,yCol):
    #global y
    global y
    y=pdf.iloc[:,[yCol]]   ### The":" after yCol transforms y into PD datafarme 
    #return Y1

set_y(df3,11) 

y.shape ### Matrice dim (10000,1)

### Créer le model avec statsmodel pour analyse

import patsy
import statsmodels.api as sm


def mymodel_anl(regType):
    global mymodel_anl
    ## Construction du Model
    #regType=" "
    #regType = input ("Quels regression : OLS / Logit ? ")
    if str(regType) == "OLS" :
        OLS_model = sm.OLS(y, sm.add_constant(X))
        result=OLS_model.fit()
        print(result.summary())
        #mymodel_anl=model
        #return mymodel_anl  ### Ou return mon model
    elif regType == "Logit" :
        logit_model = sm.Logit(y, sm.add_constant(X))
        result=logit_model.fit()
        print(result.summary())
        
### Construire le Model
        
        
mymodel_anl("OLS")


### Verifier la  Multicollinearité entre les variables explicatives

from statsmodels.stats.outliers_influence import variance_inflation_factor 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(2)

### Matrcices de Correlation 

import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

corr = X.corr()
corr

### Representation Graphique de Matrice de Correlation 
sns.heatmap(corr)
