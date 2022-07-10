#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


df = pd.read_csv ("C:\\Top Mentor\\House Price.csv")
df


# In[4]:


DV = input ('Enter the Dependent Variable: ')
position_dv = df.columns.get_loc (DV)
position_dv


# In[5]:


print (len (df))


# In[6]:


df.count()


# In[7]:


df.size


# In[8]:


df.isna().sum()


# In[9]:


df.columns


# In[10]:


#creating the loop for getting all above counts in one cell

filename = pd.DataFrame( columns= ["colname", 'misingcount', 'missingpercent', "uniquevalues"])

for col in list (df.columns.values):
    sum_missing = df[col].isnull().sum()
    perc_missing = (sum_missing/ len(df)) * 100
    uniq_values = (df.groupby ([col])[col].count()) .count()
    filename = filename.append ({"colname": col, "misingcount": sum_missing, "missingpercent": perc_missing,
                                "uniquevalues": uniq_values}, ignore_index = True)
print (filename)


# In[11]:


FileDType = (df.dtypes).reset_index()
FileDType


# In[12]:


filename = filename.merge (FileDType, left_on = "colname", right_on = "index", how = "inner")
filename


# In[13]:


del filename['index']


# In[14]:


filename = filename.rename (index = str, columns = {0 : 'DataType'})
filename


# In[15]:


df.drop (["Alley"], axis = 1, inplace = True)
df


# In[16]:


#checking if there are any outliers

def remove_outliers (df_in, col_name):
    q1 = df_in [col_name].quantile (0.25)
    q3 = df_in [col_name].quantile (0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 - 1.5 * iqr
    df_out = df_in.loc [(df_in[col_name] > fence_low)]
    return df_out    


# In[17]:


#Extracting the numeric columns

df_col = filename [(filename.DataType == 'int64') | (filename.DataType == 'float64')]
df_col


# In[ ]:





# In[18]:


for col_na in df_col ["colname"]:
    if col_na != DV:
        df = remove_outliers (df, col_na)


# In[19]:


df


# In[20]:


df.count ()


# In[21]:


df.isna().sum()


# In[22]:


# We are trying to do a groupby

df.groupby ('GarageType').agg ({'GarageType': np.size})


# In[23]:


# Filling the null values with mode
df['GarageType'] = df['GarageType'].fillna ('Attchd')


# In[24]:


df.groupby ('GarageType').agg ({'GarageType': np.size})


# In[25]:


df


# In[26]:


df.describe()


# In[27]:


df.dtypes


# In[28]:


# changing the datatype of "GarageType" (object) to the category

df["GarageType"] = df["GarageType"].astype ('category')


# In[29]:


#changed theb datatype to category
# because we cannot use labelencoding on object column
df.dtypes


# In[30]:


# label encoding the "GarageType" column

df["GarageType"] = df ["GarageType"].cat.codes


# In[31]:


# label changed to the numbers

df.groupby ('GarageType').agg ({'GarageType': np.size})


# In[ ]:





# In[32]:


# one hot encoding to the "GarageType" column

df = pd.get_dummies (df, columns = ["GarageType"], drop_first = True)


# In[33]:


# filling the missing values to "MasVnrArea" 

df["MasVnrArea"]. fillna(df["MasVnrArea"].median(), inplace = True)


# In[ ]:





# In[34]:


df


# In[ ]:





# In[35]:


dx = df["SalePrice"]
dx


# In[36]:


df.drop (["SalePrice"], axis =1, inplace =True)


# In[37]:


df


# In[38]:


df = pd.concat ([df,dx], axis = 1)
display (df)


# In[ ]:





# In[39]:


# We want to check the relationship between the columns

import seaborn as sns
sns.pairplot (df)


# In[40]:


from matplotlib.pyplot import rcParams
rcParams ['figure.figsize'] = 20, 10
df.corr ()
sns.heatmap (df.corr(), annot = True)


# In[41]:


# Spliting the data between X and Y

x = df.iloc [:,1:13]
x


# In[42]:


y = df.iloc [:, 13:]
y


# In[43]:


# Train test split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 42)


# In[44]:


# Creating the model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm = lm.fit(x_train,y_train)   


# In[45]:


# Displaying the coefficients

coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm.coef_))], axis = 1)
print (coefficients)


# In[46]:


lm.intercept_


# In[47]:


# Predicting the test data

y_pred = lm.predict(x_test)
y_pred


# In[48]:


y_test


# In[49]:


# Checking the accuracy

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[50]:


# Try and apply cross validation

from sklearn.model_selection import cross_val_score
cvm= cross_val_score(lm, x, y, cv=5)
pd.DataFrame(cvm)


# In[51]:



print ('Mean', cvm.mean())
print ('STD',cvm.std())


# In[52]:


## let's add an intercept (beta_0) to our model

import statsmodels.api as sma
x_train = sma.add_constant(x_train) 
x_test = sma.add_constant(x_test)


# In[53]:


# Applying OLS method 

import statsmodels.api as sm
lm2 = sm.OLS(y_train,x_train).fit()
lm2.summary()


# In[54]:


# we can eleminate the columns having the p_value greater then 0.05 using a function, loop or simply we can drop the columns

df = df.drop (['GrLivArea','GarageType_4'], axis = 1)
df


# In[ ]:





# In[55]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
[{x_train.columns[j]:variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])}]


# In[ ]:





# In[56]:


# Eliminating the columns having VIF value > 5 with help of function 

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no: ", i)
        print(vif, '\n')
        a = np.argmax(vif)
        print("Max VIF is for variable no:",a)
        print("Column Name:",x.columns[a], '\n')
      
        if vif[a] <= thresh :
            break
        if i == 1 :          
            x= x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
        elif i > 1 :
            x = x.drop(x.columns[a],axis = 1)
            vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    return(x)


# In[57]:


train_out = calculate_vif(x_train)


# In[58]:


x_train.shape


# In[59]:


train_out.head()


# In[60]:


# Creating the model again after removing the insignificant columns

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm = lm.fit(train_out,y_train)   


# In[61]:


import statsmodels.api as sm
lm2 = sm.OLS(y_train,train_out).fit()
lm2.summary()


# In[ ]:




