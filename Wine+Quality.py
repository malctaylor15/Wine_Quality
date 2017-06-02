
# coding: utf-8

# # Wine Quality Data Set 
# 
# Detecting the wine quality using physicochemical (inputs) and sensory (the output) variables

# In[1]:

import pandas as pd 
import pprint
import numpy as np
import sklearn as sk 
import matplotlib.pyplot as plt
import seaborn as sns 
import math 
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)


# ### First start by looking at the data set 

# In[2]:

red_wine_raw = pd.read_csv("winequality-red.csv",sep = ';' )
red_wine_raw.head(10)


# In[3]:

red_wine_raw[red_wine_raw["citric acid"] == 0 ].count()


# In[ ]:



Data available at 

https://archive.ics.uci.edu/ml/datasets/Wine+Quality


# In[4]:

red_wine_raw.describe()


# In[5]:

corr_mat = red_wine_raw.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=.8, square=True)


# ### Quick Rundown 
# 
# Strong correlations: 
# 1. Fixed acidity, density
# 2. Fixed acidity, pH 
# 3. Free sulfur dioxide, Total sulfur dioxide (expected) 
# 4. Density, alcohol 
# 

# In[6]:

col_names = list(red_wine_raw)


# In[7]:

red_wine_raw.hist()
plt.show()


# ### A Quick Variable Run down
# 
# 1. Alcohol has a nice balanced distribution. Fewer observations with higher alcohol content. Peak at 9/10 
# 2. Chlorides is pretty sparse, mainly around 0, 0.1. Probably won't be very useful 
# 3. Citric acid has a nice distribution 
# 4. Density looks normally distributed- could be a good variable 
# 5. Fixed acidity has a nice, skewed normal distribution 
# 6. Free sulfur dioxide has fewer observations as the dioxide increases.  
# 7. Sulphates nice distribution, could be normal 
# 8. Quality - categorial, strong peaks at 5 and 6. 
# 9. Volatile acidity - Peak at 0, tapering off quickly 
# 10. Sulphates- looks as if there is a peak around the mean has a decent distribution 
# 11. Total sulfur dioxide is bounded by 0 and decreasing as dioxide increases- nice distribution 
# 12. Volatile acidity- nice distribution, peaks from 0.4- 0.6ish 

# In[8]:

red_wine_raw.plot(kind = 'density',subplots = True, layout =(3,4), sharex = False )
plt.show()


# ### Whisker Plots 

# In[9]:

red_wine_raw.plot(kind = 'box', subplots = True, layout = (3,4), sharex = False, sharey = False)
plt.show()


# *Interpret these graphs*
# 
# -- Remember these have to do mainly with quantiles 
# Quality is skewed, chlorides and residual sugars have low variance and are tightly distributed- maybe have many values around peaks 

# In[10]:

from pandas.tools.plotting import scatter_matrix


# In[11]:

scatter_matrix(red_wine_raw)
plt.show()


# Not sure how to interpret this. 
# Looking for linear patterns but hard to detect due to high number of data points 
# Many of the plots have trends but due to categorical nature of quality, hard to see scatter plots clearly 
# 

# ### Let's try making some features that could be helpful 

# In[12]:

red_wine_raw.columns.values


# In[13]:

red_wine_edit = red_wine_raw


# In[14]:

red_wine_edit["free_sulf_ratio"] = red_wine_raw["free sulfur dioxide"]/ red_wine_raw["total sulfur dioxide"]
red_wine_edit["fixed_volatile_perc"] = red_wine_raw["fixed acidity"]/(red_wine_raw["volatile acidity"] + 
                                                                      red_wine_raw["fixed acidity"])
red_wine_edit["sugar_alcohol_ratio"] = red_wine_raw["residual sugar"]/ red_wine_raw["alcohol"]
red_wine_edit["sugar_density_ratio"] = red_wine_raw["residual sugar"]/ red_wine_raw["density"]
red_wine_edit["chlorides_sulphates_ratio"] = red_wine_raw["chlorides"]/red_wine_raw["sulphates"]


# In[15]:

num_indep_vars = len(red_wine_edit.columns.values)
indep_var_names = red_wine_edit.columns.values
print("There are now ", (num_indep_vars), " independent variables. They are \n" ,indep_var_names)


# ## Pre Processing

# Let's make a standardized and normalized copy of the data  
# We will import the relevant pacakges and then make 2 copies of the data set 

# In[16]:

from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 


# In[17]:

scaler = MinMaxScaler(feature_range = (0,1)).fit(red_wine_edit)
red_wine_scale = scaler.transform(red_wine_edit)
pd.DataFrame(red_wine_scale).describe()


# Note the min is 0 and max is 1 and we can potentially sort by standard deviation to under relatively which variable has the most variance 
# 
# Also we can compare the quantiles and see that most of the 75% quantiles are below 0.6 so there might be some outliers. 

# In[18]:

red_wine_standard_scaler = StandardScaler().fit(red_wine_edit)
red_wine_standard = red_wine_standard_scaler.transform(red_wine_edit)
pd.DataFrame(red_wine_standard).describe().round(4)


# Note the Mean and std being 0 and 1 relatively. 
# 
# Let's use the Standardized data because we will start with some linear regression stuff 

# # Feature Initial Selection 

# Use KBest variables, univariate selection and PCA to get an idea of the best variables 

# Use Recursive Feature elimination to find ideal number of variables for linear regression 

# In[19]:

red_wine_edit.columns.values


# In[20]:

X = np.delete(red_wine_scale,11,1)
Y = red_wine_scale[:,11]


# In[ ]:




# In[21]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[22]:

model1 = LinearRegression()
rfe1 = RFE(model1, 4)
fit1 = rfe1.fit(X,Y)


# In[23]:

print("Num Features: ", fit1.n_features_)
print("Selected Features: ", fit1.support_)  
print("Feature Ranking: ", fit1.ranking_) 


# In[ ]:




# We see that the ideal number of features for the linear regression is 4 as compared to the original 11. 
# Those variables are : 
# 
# If we wanted to reduce the variance or computation time for the other processes we could use these variables. 
# It will be interesting to compare variable importances in the other algorithms. 

# # Begin Preparing Data for Algorithms 

# Split data into test and train sets 

# In[24]:

from sklearn.model_selection import train_test_split
# Remember that quality is the 12th column of the dataset 
Y = red_wine_standard[:,11]
red_wine_standard2 = np.delete(red_wine_standard, 11, axis = 1)
X_train, X_test, y_train, y_test = train_test_split(red_wine_standard2, Y, test_size = 0.25)


# In[25]:

X_train.shape


# # Begin using the Algorithms 

# ### Start with Linear regression 

# In[26]:

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# In[27]:

linear_fit = LinearRegression()
linear_fit.fit(X_train, y_train)


# In[28]:

linear_fit.coef_


# In[29]:

coef_dict1 ={}
x = 0
for data in red_wine_edit.columns.values[0:(num_indep_vars-1)]:
    coef_dict1[data] = linear_fit.coef_[x]
    x = x+1 

pprint.pprint(coef_dict1)


# In[30]:

# Absolute value of the coefficients 
coef_abs_dict1 ={}
x = 0
for data in red_wine_raw.columns.values[0:(num_indep_vars-1)]:
    coef_abs_dict1[data] = np.absolute(linear_fit.coef_[x])
    x = x+1 

pprint.pprint(coef_abs_dict1)


# These are scaled coefficients so we should not interpret them with the variable associated but rather against each other. We can see that alcohol and the hig

# In[31]:

linear_fit.score(X_train, y_train)


# In[32]:

linear_fit.score(X_test, y_test)


# ### Random Forest 

# In[33]:

from sklearn.ensemble import RandomForestRegressor


# In[34]:

rf1 = RandomForestRegressor().fit(X_train, y_train)


# In[35]:

rf1.score(X_test, y_test)


# In[36]:

rf1.score(X_train, y_train)


# In[37]:

# specify parameters and distributions to sample from
param_dist = {"n_estimators": [10, 30, 100, 200, 400], 
              "max_depth": [1,3, 6, None],
              "min_samples_split": [2,3,7,13],
              "min_samples_leaf": [2,3,7,11], 
              "max_leaf_nodes": [2,4, None],
              "max_features": [None, 'auto', 'sqrt', 3, 7, 11]
              }
n_iter_search = 15


# In[38]:

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# We won't do grid search because it will take too long. 

# In[39]:

rf1_random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist,
                                   n_iter=n_iter_search)


# In[40]:

rf1_random_search.fit(X, Y)


# In[41]:

pprint.pprint(rf1_random_search.cv_results_)


# In[42]:

rf1_random_search.best_score_


# In[43]:

rf_best = rf1_random_search.best_estimator_


# In[44]:

rf_best


# In[45]:

rf_imp1 ={}
x = 0
for data in red_wine_raw.columns.values[0:(num_indep_vars-1)]:
    rf_imp1[data] = rf_best.feature_importances_[x]
    x = x+1 

pprint.pprint(rf_imp1)


# We can see that the most important are 
# * alcohol 
# * sulphates 
# * volatile acidity 
# * total sulfur dioxide (maybe) 

# ### Evaluation Metric 

# In[46]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits= 7, random_state=22)


# In[47]:

lin_results = cross_val_score(linear_fit, X, Y, cv=kfold)
print((lin_results.mean()*100.0, lin_results.std()*100.0))


# In[48]:

rf_results = cross_val_score(rf_best, X, Y, cv = kfold)
print(rf_results.mean()*100.0, rf_results.std()*100.0)


# Note that the random forest has a higher mean and lower standard deviation. 
# We can be fairly certain that the random forest will perform better than a linear regression 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



