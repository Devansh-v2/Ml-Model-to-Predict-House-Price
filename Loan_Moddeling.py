'''
ID : Customer ID

Age : Customer's age in completed years

Experience : #years of professional experience

Income : Annual income of the customer ($000)

ZIP Code : Home Address ZIP code.

Family : Family size of the customer

CCAvg : Avg. spending on credit cards per month ($000)

Education : Education Level.
1: Undergrad;
2: Graduate;
3: Advanced/Professional

Mortgage : Value of house mortgage if any. ($000)

10.Personal Loan : Did this customer accept the personal loan offered in the last campaign?

11.Securities Account : Does the customer have a securities account with the bank?

12.CD Account : Does the customer have a certificate of deposit (CD) account with the bank?

13.Online : Does the customer use internet banking facilities?

14.Credit card : Does the customer use a credit card issued by
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn import tree


df=pd.read_csv("Bank_Personal_Loan_Modelling.csv")
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())
print(df.describe())

df.drop(["ID"],axis=1,inplace=True)
print("No.Of Unique Values:- \n-------------------------\n",df.nunique(),sep='')

# Family, Education,Personal Loan,Securities Account,CD Account,Online,CreditCard are Categorical Values

cat_cols=["Family",'Education','Personal Loan','Securities Account','CD Account','Online','CreditCard']
cont_columns=['Age','Experience','Income','CCAvg','Mortgage']

# Pre-Processing

# Viewing the negative experience Values

print(df[df['Experience']<0].groupby(["Age","Education"])["Experience"].count())
df['Experience']=df['Experience'].apply(lambda x:np.abs(x)if x<0 else x)

# EDA

for cont_columns in cat_cols:
    print("Feature:",cont_columns)
    print(df[cont_columns].value_counts().sort_index())
    print()
    print("#"*20)
    print()

def plot_pie_counts(cont_columns,r,c,n):
    plt.subplot(r,c,n)
    df[cont_columns].value_counts().plot.pie(autopct='%1.1f%%')
    plt.subplot(r,c,n+1)
    sns.countplot(x=cont_columns,data=df,hue='Personal Loan')


for i,c in enumerate(cat_cols):
    plot_pie_counts(c,3,6,(i*2)+1)


plt.figure(figsize=(25, 15))
plt.show()


sns.heatmap(df.corr(),cmap='BrBG',annot=True)
plt.show()

# ML

x=df[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']]
y=df['Personal Loan']

xtrain,xtest,ytain,ytest=train_test_split(x,y,random_state=54,test_size=0.25)

model_log=LogisticRegression().fit(xtrain,ytain)
model_dt=DecisionTreeClassifier().fit(xtrain,ytain)
model_rf=RandomForestClassifier().fit(xtrain,ytain)
model_svc=SVC().fit(xtrain,ytain)


print("Accuracy of Model_Log is:")
print(round(100*model_log.score(xtest,ytest),2))

print("Accuracy of Model_DT is:")
print(round(100*model_dt.score(xtest,ytest),2))

print("Accuracy of Model_RF is:")
print(round(100*model_rf.score(xtest,ytest),2))

print("Accuracy of Model_SVC is:")
print(round(100*model_svc.score(xtest,ytest),2))


coef=pd.DataFrame(model_log.coef_,index=['Coefficient'],columns=x.columns).T
print(coef)
print()

sns.set_style("whitegrid")


def performance(xt,yt,model):
    yp=model.predict(xt)
    print('Classification Report')
    print('-'*30)
    print("R2 Score:",model.score(xt,yt))
    print((classification_report(yt,yp)))
    print('-' * 30)
    plt.subplot(1,2,1)
    sns.heatmap(pd.DataFrame(confusion_matrix(yt,yp),index=['No','Yes'],columns=['No','Yes']),annot = True ,fmt='d')
    plt.title("Confusion Matrix")
    rocAuc=roc_auc_score(yt,yp)
    fpr,tpr,th=roc_curve(yt,yp)
    plt.subplot(1,2,2)
    plt.plot(fpr,tpr,label='Model(area=%0.2f)'%rocAuc)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.show()

print("For Logistic Regression")
performance(xtest,ytest,model_log)

print("For Decision Tree")
performance(xtest,ytest,model_dt)

print("For Random Tree")
performance(xtest,ytest,model_rf)

print("For SVC")
performance(xtest,ytest,model_svc)


# Building The Tree

def build_tree(model):
    plt.figure(figsize=(15,15))
    out=tree.plot_tree(model,feature_names=x.columns,filled=True,fontsize=10,node_ids=True,class_names=True)
    for p in out:
        arrow=p.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor=('black')
            arrow.set_linewidth(1)

    plt.show()


build_tree(model_dt)

def show_importance(model):
    importance=model.feature_importances_
    idx=np.argsort(importance)
    plt.title("Feature Importance")
    plt.barh(range(len(idx)),importance[idx],color="purple",align='center')
    plt.yticks(range(len(idx)),[x.columns[i] for i in idx])
    plt.xlabel('Importance')
    plt.show()

show_importance(model_dt)

params_dt={"criterion":['gini','entropy'],
           "min_samples_leaf":[1,2,5,7,11,15,20,25],
           "max_depth":np.arange(1,10),
           "max_leaf_nodes":[5,10,15,20,25,30,35]}

model_grid_dt1=GridSearchCV(DecisionTreeClassifier(),param_grid=params_dt,).fit(xtrain,ytain)
print(model_grid_dt1.best_params_)


# Improved Tree

build_tree(model_grid_dt1.best_estimator_)
show_importance(model_grid_dt1.best_estimator_)

# Improved Model

x2=df[['Income','Family','Education','CD Account','CCAvg']]
y2=df['Personal Loan']

xtrain2,xtest2,ytrain2,ytest2=train_test_split(x2,y2,random_state=89,test_size=0.15)


model_rf2=RandomForestClassifier().fit(xtrain2,ytrain2)

print("Accuracy of The Improved Random Forest is:")
print(round(100*model_rf2.score(xtest2,ytest2),2))

print("For Improved RF Model")
performance(xtest2,ytest2,model_rf2)

ypred=model_rf2.predict(xtest2)
print(xtest2.head())

print("Predicted Values")
print(ypred[:5])

print("Original Values")
print(ytest2[:5].values)

print("RF is Better ")


# Saving ML Model


pd.to_pickle(model_rf2,"Loan Predictor.pkl")
m=pd.read_pickle("Loan Predictor.pkl")


# Final Model


income=eval(input("Enter Annual income of the customer ($000):"))
fam=eval(input("Enter Family size of the customer:"))
edu=eval(input("Enter Education Level:- 1: Undergrad, 2: Graduate, 3: Advanced/Professional:"))
cda=eval(input("Enter Does the customer have a certificate of deposit (CD) account with the bank? (0:No,1:Yes):"))
cca=eval(input("Enter Avg. spending on credit cards per month ($000):"))

query=pd.DataFrame({'Income':[income],'Family':[fam],'Education':[edu],"CD Account":[cda],'CCAvg':[cca]})

if m.predict(query) == 0:
    print('Not a Potential Customer')

else:
    print("Potential Customer")
