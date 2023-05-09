#!/usr/bin/env python
# coding: utf-8

# # Predicting the sale price of bulldozers using ml
# 
# ### Problem definition
# > Predicting the price of a bulldozer given the characteristics and previous example 
#   of how much similar bulldozer have been sold for.
# 
# ### Data
# > Data is downloaded from : https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview
# 
# > The data for this competition is split into three parts:
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set     throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 -   November 2012. Your score on the test set determines your final rank for the competition.
# 
# > The key fields are in train.csv are:
# * SalesID: the uniue identifier of the sale
# * MachineID: the unique identifier of a machine.  A machine can be sold multiple times
# * saleprice: what the machine sold for at auction (only provided in train.csv)
# * saledate: the date of the sale
# 
# ### Evaluation
# > The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction   prices.
#   
#   check:https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview/evaluation
#   
#   The goal of the regression evaluation is to minimize the the error.
# 
# ### Features
# > Check out the data dictionary here:https://www.kaggle.com/competitions/bluebook-for-bulldozers/data
# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[141]:


# Import training and validation set
#df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv", low_memory=False)


# In[142]:


#df.info()


# In[143]:


#df.isna().sum()


# In[144]:


# df.head()


# In[145]:


#df["saledate"].dtype # pandas string


# In[146]:


#df.columns


# In[147]:


# fig, ax = plt.subplots() # subplots-mulitple axes and subplot - one axes
# ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[148]:


# df.SalePrice.plot.hist()


# ### Parsing dates
# Changing data type from pandas string to datetime

# In[149]:


# Import data again but parse saledate column this time
df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])


# In[150]:


df.saledate.dtype


# In[151]:


df.saledate[:10]


# In[152]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000]);


# In[153]:


df.head()


# In[154]:


df.head().T


# ### sort dataframe by date
# when working with time series data, it's a good idea to sort it by date.

# In[155]:


df.sort_values(by=["saledate"], inplace=True, ascending=True)


# In[156]:


df.head(10)


# In[157]:


# Make a copy of original data frame
df_tmp = df.copy()


# ### Add date time parameter for saledate column

# In[158]:


df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saledayofWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saledayofYear"] = df_tmp.saledate.dt.dayofyear


# In[159]:


df_tmp.head().T


# In[160]:


# Now we have enrich our data frame with date, time features, we can remove saledate column 
df_tmp.drop("saledate", axis=1, inplace=True)


# In[161]:


df_tmp.state.value_counts()


# In[162]:


df_tmp.info()


# ## Convert string dtype to categories
# One way to do that is by converting them to pandas categories

# In[163]:


pd.api.types.is_string_dtype(df_tmp.state)


# In[164]:


# Check which columns are string dtype


# In[165]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
       print(label)


# In[166]:


for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[167]:


df_tmp.info()


# In[168]:


df_tmp.state.cat.categories


# In[169]:


df.state.value_counts()


# In[170]:


df_tmp.state.cat.codes


# Now we have access to all our data in the form of numbers.

# In[171]:


#Check missing data
df_tmp.isnull().sum()/len(df_tmp)


# ## Save processed data

# In[1]:


# df_tmp.to_csv("Data/bluebook-for-bulldozers/train_tmp2.csv",
#               index=False)


# In[171]:


#Import  data
df_tmp = pd.read_csv("Data/bluebook-for-bulldozers/train_tmp2.csv",
                    low_memory=False)


# In[172]:


df_tmp.head().T


# In[173]:


pd.Categorical(df_tmp.state).codes


# In[174]:


# Adding 1 so there is no -1 for null values
# pd.Categorical(df_tmp["UsageBand"]).codes + 1


# In[175]:


df_tmp.info()  


# ### SPLIT THE DATA INTO TRAIN AND VAL

# In[176]:


# train and val splits
df_val = df_tmp[df_tmp.saleYear == 2012]

df_train = df_tmp[df_tmp.saleYear != 2012]


# In[177]:


df_train.head()


# In[178]:


df_tmp[df_tmp["saleYear"]==2012]


# In[179]:


df_val.head()


# In[180]:


#Turn Categorical variable into numbers and fill missing
def missing_cat(df):
    for label, content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add a binary column to indicate wether sample has numeric value
            df[label+"_is_missing"] = pd.isnull(content)
            #Turn categories into numbers and add + 1
            df[label] = pd.Categorical(content).codes + 1
    return df        


# In[181]:


df_val = missing_cat(df_val)


# In[182]:


df_train = missing_cat(df_train)     


# In[183]:


df_val.head()


# In[184]:


df_train.head()


# ### Fill missing numerical values

# In[185]:


for label, content in df_train.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[186]:


for label, content in df_val.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[187]:


# Fill the numeric rows with the median
def missing_num(df):
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                #Add a binary column 
                df[label+"_is_missing"] = pd.isnull(content)
                #Fill missing value with median
                df[label] = content.fillna(content.median()) 
    return df            


# In[188]:


# We did not chose mean because mean is sensitive to outliers


# In[189]:


df_train = missing_num(df_train)
df_val = missing_num(df_val)


# In[190]:


df_train.isna().sum()


# In[191]:


df_val.isna().sum()


# In[192]:


df_val["auctioneerID_is_missing"]=False

#Rearranging the columns order
df_val = df_val[df_train.columns]


# ### Split into X_train and y_train and X_val and y_val

# In[193]:


X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_val, y_val = df_val.drop("SalePrice", axis=1), df_val["SalePrice"]


# In[194]:


X_train.head()


# ## Building evaluation function

# In[195]:


from sklearn.metrics import mean_squared_log_error, mean_absolute_error

def rmsle(y_test, y_preds):
    """
    rmsle - Root Mean Squared Log error
    
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Function for other evaluation metrices
def show_score(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    scores={"Training MAE": mean_absolute_error(y_train, train_preds),
            "valid MAE":mean_absolute_error(y_val, val_preds),
            "training RMSLE": rmsle(y_train, train_preds),
            "val RMSLE":rmsle(y_val, val_preds),
            "Train R^2":sklearn.metrics.r2_score(y_train, train_preds),
            "valid R^2":sklearn.metrics.r2_score(y_val, val_preds)}
    return scores


# ## Testing our model on a subset 

# In[196]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,
                              max_samples=10000,
                              random_state=3)

model.fit(X_train, y_train)

scores = show_score(model)

scores


# ## Hyperparameter tuning with RandomizedSearchCV

# In[197]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.model_selection import RandomizedSearchCV\n\n# grid\nrf_grid ={"n_estimators": np.arange(10, 100, 10),\n          "max_depth": [None, 3, 5, 10],\n          "min_samples_split": np.arange(2,20,2),\n          "min_samples_leaf": np.arange(1, 20, 2),\n          "max_features":[0.5, 1, "sqrt", "auto"],\n          "max_samples":[10000]}\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                    random_state=3),\n                              param_distributions=rf_grid,\n                              n_iter=2,\n                              cv=5,\n                              verbose=True)\n\nrs_model.fit(X_train, y_train)\n')


# In[198]:


rs_model.best_params_


# In[199]:


show_score(rs_model)


# #### NOTE: These were found after 100 iterations of RandomizedSearchCV.

# In[200]:


get_ipython().run_cell_magic('time', '', '# Most ideal hyperparameters\nideal_model =RandomForestRegressor(n_jobs=-1,\n                                   n_estimators=40,\n                                   min_samples_leaf=1,\n                                   min_samples_split=14,\n                                   max_features=0.5,\n                                   max_samples=None)\n\nideal_model.fit(X_train, y_train)\n')


# In[201]:


show_score(ideal_model)


# In[211]:


## Import test data
df_test = pd.read_csv("Data/bluebook-for-bulldozers/Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])


# In[212]:


df_test.columns


# In[213]:


df_test.info()


# In[214]:


set(X_train.columns) - set(df_test.columns)


# In[215]:


## Preprocess test data
def preprocess(df):
    
    
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saledayofWeek"] = df.saledate.dt.dayofweek
    df["saledayofYear"] = df.saledate.dt.dayofyear
    
    for label, content in df_tmp.items():
        if pd.api.types.is_string_dtype(content):
            df_tmp[label] = content.astype("category").cat.as_ordered()
    
    df.drop("saledate", axis=1, inplace=True)
    
    df = missing_num(df)
    df = missing_cat(df)
    
    df["auctioneerID_is_missing"]=False
    
    df=df[X_train.columns]
    
    return df


# In[216]:


df_test = preprocess(df_test)


# In[217]:


test_preds = ideal_model.predict(df_test)


# In[219]:


test_preds


# In[220]:


df_preds = pd.DataFrame()
df_preds["Sales_ID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[222]:


df_preds.to_csv("Data/bluebook-for-bulldozers/df_preds.csv", index=False)


# #### Feature Importance

# In[ ]:




