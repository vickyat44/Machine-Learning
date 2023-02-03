import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style="darkgrid")
df1 = pd.read_csv("diabetes2.csv") 
 
df1.info() 
df1.describe() 
print("Proportion of missing values")
missing_percentage = (df1==0).sum()*100/df1.shape[0] 
bp_df = df1.loc[df1['SkinThickness']==0] 
print("Count of zeros in blood_pressure:", (bp_df['BloodPressure']==0).sum()) 
print("Count of zeros in skinfold_thickness:", (bp_df['SkinThickness']==0).sum()) 
print("Count of zeros in insulin:", (bp_df['Insulin']==0).sum()) 
bp_df = df1.loc[df1['Insulin']==0] 
print("Count of zeros in blood_pressure:", (bp_df['BloodPressure']==0).sum()) 
print("Count of zeros in skinfold_thickness:", (bp_df['SkinThickness']==0).sum()) 

print("Count of zeros in insulin:", (bp_df['Insulin']==0).sum()) 
m_col = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'] 
colnum=1 

for col in m_col: 
    df1[col]=df1[col].replace(0,np.nan) 
    colnum+=1 
df1.isnull().sum()/df1.shape[0] 

from sklearn.impute import KNNImputer 
imputer = KNNImputer(n_neighbors=14) 
k_df1=imputer.fit_transform(df1) 
df=pd.DataFrame(k_df1,columns=df1.columns)
 
df.isnull().sum() 
df['Outcome'].value_counts().plot.pie() 
df['Outcome'].value_counts(normalize=True)
N_col = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
plt.figure(figsize=(20,60), facecolor='white')
plotnum=1 
for col in N_col: 
    ax=plt.subplot(9,3,plotnum) 
    sns.boxplot(y=df[col],x=df['Outcome'])
    plt.title(col) 
    plotnum+=1 
    
plt.show() 
plt.figure(figsize=(20,60), facecolor='white') 
plotnum=1 
for col in N_col: 
    ax=plt.subplot(9,3,plotnum) 
    sns.histplot(df[col], bins=20) 
    plt.title(col) 
    plotnum+=1 
    
plt.show() 
plt.figure(figsize=(10,7)) 
sns.heatmap(df.corr(),annot=True, cmap='Reds',center = 0.4) 
plt.title('Correlation of Different columns in dataframe') 
plt.show() 
X=df.drop(['Outcome'],axis=1) 
y=df.Outcome 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve,auc, plot_confusion_matrix 
    
log_model = LogisticRegression(solver='liblinear') 
log_model.fit(X_train,y_train) 
y_pred = log_model.predict(X_test) 
res= confusion_matrix(y_test, y_pred) 
sns.heatmap(res/np.sum(res), annot=True, fmt='.2%') 
plt.xlabel('Actual label') 
plt.ylabel('Predicted label') 
plt.show() 

def Model_Performance(test, pred): 
    precision = precision_score(test, pred)
    recall = recall_score(test, pred) 
    f1 = f1_score(test, pred) 
    print('1. Confusion Matrix:\n',confusion_matrix(test, pred)) 
    print("\n2. Accuracy Score:", round(accuracy_score(test, pred)*100,2),"%") 
    print("3. Precision:", round(precision*100,2),"%") 
    print("4. Recall:",round(recall*100,2),"%" ) 
    print("5. F1 Score:",round(f1*100,2),"%" )
    print("6. clasification report:\n",classification_report(test, pred)) 
    Model_Performance(y_test, y_pred) 
    fpr, tpr, _ = roc_curve(y_test, y_pred) 
    roc_auc = auc(fpr, tpr) 
    plt.figure() 
    plt.plot(fpr, tpr, label='ROC curve(area= %2.f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlim([0.0, 1.0]) 
    plt.ylim([0.0, 1.05]) 
    plt.xlabel('False positive rate') 
    plt.ylabel('True positive rate') 
    plt.title('ROC curve') 
    plt.legend(loc='lower right') 
    plt.grid() 
    plt.show() 
    from sklearn.metrics import roc_auc_score 
    print('AUC Score of Model:', round(roc_auc_score(y_test, y_pred) * 100, 2), "%") 
    model_f = pd.DataFrame() 
    model_f['Features'] = list(X_train.columns) 
    model_f['importance'] = list(log_model.coef_[0]) 
    imp_check = pd.DataFrame(model_f.sort_values(by='importance')) 
    imp_check.plot.barh(x='Features', y='importance', title='Features by Importance') 
    plt.show() 
    print(imp_check)