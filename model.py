import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# dataset = pd.read_csv('sales.csv')

# dataset['rate'].fillna(0, inplace=True)

# dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()

# regressor.fit(X, y)





# second model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas_profiling import ProfileReport
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


Data = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
Data


data = pd.DataFrame(Data)
data


# profile = ProfileReport(data, title="car Data Report")
# profile

data.describe()

del data['Car_Name']

Age = 2023 - data['Year']
data.insert(1,'Age',Age)

del data['Year']
data

x = pd.DataFrame(data, columns = ['Age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']) 
y = data['Selling_Price'].values.reshape(-1,1)

x_train, x_test, y_train,y_test = train_test_split(x,y , test_size = 0.2 , random_state = 0)

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2score:",metrics.r2_score(y_test,y_pred))

# creating new feature

x =data.drop('Selling_Price',axis = 1)
y = data['Selling_Price'].values.reshape(-1,1)

def check(dimension,testsize):
    r2 = 0.8793462370195231
    for column in x:
        new_column_name = column + str(dimension)
        new_column_val = x[column]**(dimension)
        x.insert(0 , new_column_name , new_column_val)
        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = testsize, random_state = 0)
        regressor = LinearRegression()
        regressor.fit(x_train,y_train)
        y_pred = regressor.predict(x_test)
        r2_new = metrics.r2_score(y_test , y_pred)
        if r2_new < r2:
            x.drop([new_column_name],axis =1 ,inplace =True)
            print(r2_new)
        else:
            r2 = r2_new
    print('r2 score',r2)
    
    
    
    
check(2,0.2)    


profile2 = ProfileReport(x, title="car Data Report")
profile2

pres_Kms = x['Present_Price']*x['Kms_Driven']
pres_Kms2 = x['Present_Price']*x['Kms_Driven2']
pres_Age = x['Present_Price']*x['Age']
pres_Age2 = x['Present_Price']*x['Age2']
pres_Trans = x['Present_Price']*x['Transmission']
pres_Trans2 = x['Present_Price']*x['Transmission2']
pres2_Trans = x['Present_Price2']*x['Transmission']
pres2_Trans2 = x['Present_Price2']*x['Transmission2']
pres2_Age = x['Present_Price2']*x['Age']
pres2_Age2 = x['Present_Price2']*x['Age2']
pres2_Kms = x['Present_Price2']*x['Kms_Driven']
pres2_Kms2 = x['Present_Price2']*x['Kms_Driven2']


x.insert(0,'pres_Kms',pres_Kms)
x.insert(0,'pres_Kms2',pres_Kms2)
x.insert(0,'pres_Age',pres_Age)
x.insert(0,'pres_Age2',pres_Age2)
x.insert(0,'pres_Trans',pres_Trans)
x.insert(0,'pres_Trans2',pres_Trans2)
x.insert(0,'pres2_Trans',pres2_Trans)
x.insert(0,'pres2_Trans2',pres2_Trans2)
x.insert(0,'pres2_Age',pres2_Age)
x.insert(0,'pres2_Age2',pres2_Age2)
x.insert(0,'pres2_Kms',pres2_Kms)
x.insert(0,'pres2_Kms2',pres2_Kms2)

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size= 0.2, random_state = 0)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print('r2score',metrics.r2_score(y_test, y_pred))







pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500]]))