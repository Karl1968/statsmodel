# -*- coding: utf-8 -*-
"""
Date : 05.03.2019
Spyder Editor
File : my_stats_model.py
Visualizer les model statistique d'un dataset

"""

import numpy as np
import pandas as pd


### Importer le dataset
def get_csv():
    global data
    response = input("Saisir le nom du fichier ? ")
    file=response
    dataset = pd.read_csv(file)
    data=dataset
    print ("Les 5 lignes du dataset")
    print(dataset.head(5))


get_csv()


def df_preproc(data):
    
### Faire une description statistique du Dataset
    print ("#### Description statistique du Dataset")
    print(data.describe(include='all'))

### Lister les index de colonnes 
    cnt = 0
    print("### Indexes de colonnes")
    for p in data.columns:
        print (cnt,p)
        cnt=cnt+1
        
### Gestion des données manquantes
    ### Afficher les données manquantes
    print ("#### Afficher les données manquantes")
    print(data.isna().sum())
    
    
### Afficher les variables categoriques "type object"
    print ("#### Afficher les variables categoriques type object")
    print(data.dtypes)
    
df_preproc(data)



### traiter les valeurs manquantes


def missValues(data):
    print ("Dataset Option 1 ( Strategy = Mean) Option 2 (Strategy = median")
    response = input("Please enter option: 1 or 2:  ")
    strategy=int(response)
    global df2
    if strategy == 1:
        df2=data.fillna(data.mean()).copy()
        return df2
    elif strategy ==2:
        df2=data.fillna(data.median()).copy()
        return df2
    
    
missValues(data)


### Traiter les dummies Variables 

    
def treat_dum():
    global df2
    print ("Dataset name et index Colonne avec valeurs binaires à traiter")
    idxCol = input(" Index de la colonne :")
    xCol=int(idxCol)
    print (data_dum, xCol)
    dum_col=df2.iloc[:,[xCol]].columns
    df2=pd.get_dummies(df2, columns=dum_col)
    save_df(df2)
    return(df2)
    
    
(treat_dum(df2,0))


### Transformer dummy variable en binaire 
### Convertir la colonne en array
def convert_col(pdf,xCol):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    global y
    y=pdf.iloc[:,xCol].values
    y = labelencoder_y.fit_transform(y)
    return(y)
    #print(y.dtype)
    
convert_col(df2,2)
type(y),y.shape


### Dropper les colonnes non necessaire

def drop_col(pdf,xCol):
    global df_for_anl
    df_for_anl=pdf.drop(pdf.iloc[:,[xCol]].columns, axis=1)
    return df_for_anl

drop_col(df2,2)


### Rajouter la nouvelle variabe au dataframe

def add_dum(pdf,xCol):
    global df_for_anl
    df_for_anl["Purchase"] = xCol
    return df_for_anl

add_dum(df_for_anl,y)
    


### Preparation du dataframe pour stats model

### Choix de variables independant  "X"- insérer Col index

def set_X(pdf,xCol1,xCol2):
    global X1
    X=pdf.iloc[:,xCol1:xCol2]
    X1=X.columns.tolist().copy()
    X1=X.columns.str.cat(sep=" + ")
    return X1

set_X(df_for_anl,0,3)

### Choix de la Variable dependante "y"
def set_y(pdf,yCol):
    #global y
    global Y1
    y=pdf.iloc[:,yCol:]   ### The":" after yCol transforms y into PD datafarme 
    Y1=y.columns.str.cat() + " ~ "
    return Y1

set_y(df_for_anl,-1)

### Set The Formula
### formula = 'y ~ x1 + x2 + xn + c(xn1)'

def set_formula(Y,X):
    global formula
    formula = Y + X   
    return formula

set_formula(Y1,X1)


### Set the model 

    
def mymodel_anl(formula, dataf):
    import patsy
    import statsmodels.api as sm
    global mymodel_anl
    
    ### Avec Statsmodel y vient en premier
    y, X = patsy.dmatrices(formula, dataf)
    ## Construction du Model
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model  ### Ou return mon model

### Visualizer les statistiques du model
def show_results(formula,dataf):
    result = mymodel_anl(formula,dataf)
    print(result.summary())
        

mymodel_anl(formula,df_for_anl)

show_results(formula,df_for_anl)



