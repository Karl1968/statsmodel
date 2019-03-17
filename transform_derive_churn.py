########################################
### transformer les variables  
########################################
### Transform Balance en Log Balance
### L'index de Balance est 6

save_df3=df3.copy()

logBal=df3.iloc[:,[6]].values ## On extrait la variable pour la traiter 

logBal=np.log10(logBal+1) ### On Convert la Variable en Log avec Numpy

df3['Log_Balance'] = logBal  ### On rajoute au Dataframe

### Deriver les Variables

cnt = 0
print("### Indexes de colonnes")
for p in df3.columns:
    print (cnt,p)
    cnt=cnt+1
        
#VarSomme=df3.iloc[:,[3,5,7,8]].values ### No Need for this case

### Nouvel effet avec l'addition des variables "Tenure + NumProduct + HasCRdCard
### + IsActiveMemeber

VarSomme=df3.iloc[:,[3,5,7,8]]

ty=VarSomme.values[:,0:5].sum(axis =1)

### Additionner cete nouvelle colonne à notre Datframe

df3['NewEffect']=ty

### Nous testons le modéle avec les nouvelles variables 
### identique au Tuto de Churn Modelling

xCol1=[3,4,5,7,9,13,15,17]

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

### Graphique de Matrice de Correlation []
sns.heatmap(corr)
