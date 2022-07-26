# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading File

df=pd.read_csv("USA_Housing.csv")
print("First Five Values")
print(df.head())
print(df.describe())

# Checking For Regression

df.plot.scatter('Price','Avg. Area Income')
plt.show()

sns.pairplot(df,kind="reg")
plt.show()

sns.heatmap(df.corr(),annot=True)
plt.show()

p=df.columns
print(p)


# Feature Selection


x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Area Population']]
y=df['Price']


# Data Splitting


xtrain,xtest, ytrain,ytest=train_test_split(x,y,test_size=0.25, random_state=101)


# ML Model


model=LinearRegression().fit(xtrain,ytrain)
coef=pd.DataFrame(model.coef_,columns=['coeff'],index=x.columns)

print("Accuracy of Model is:")
print(round(100*model.score(xtest,ytest),2))

ypred=model.predict(xtest)
print(xtest.head())

print("Predicted Values")
print(ypred[:5])

print("Original Values")
print(ytest[:5].values)

plt.scatter(ytest,ypred)
plt.show()


# Saving ML Model


pd.to_pickle(model,"House Price Predictor.pkl")
m=pd.read_pickle("House Price Predictor.pkl")


# Final Model


income=eval(input("Enter Your Income:"))
house_age=eval(input("Enter Your House Age:"))
room=eval(input("Enter Your No.of Rooms:"))
pop=eval(input("Enter Population in your area:"))

query=pd.DataFrame({'Avg. Area Income':[income],'Avg. Area House Age':[house_age],'Avg. Area Number of Rooms':[room],"Area Population":[pop]})


print(round(m.predict(query)[0],2))