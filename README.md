# Wine_Quality

In this notebook we use Bayesian Analysis to understand what qualities make a red wine good or bad.  
The dataset is from UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
The dataset contains 12 variables that include chemical information about the wine and a quality. 
Quality is on a 1-10 scale and is categorical. 

## EDA 
In the first few steps, we begin looking at the distributions and histograms of all the variables. We can undertstand the range and more about the distributions of the data. 

## Dependent Variable Transformation 

We make the quality variable into a binary variable by splitting on the mean which is about 5.6.  
When we make the column binary, we create a new column that is 0 where the quality is 5 or below and  
the column is 1 when quality is 6 or greater. This split gives 744 "0"s or low quality wines and 855 "high" quality wines. 
The ratio of high quality to low quality wines is around 1 which is good for now. 


## Logistic Regression  

Before we go into Bayesian analysis, we conduct a standard logistic regression.  
We wanted to reduce the number of variables and model complexity before doing the Bayesian analysis.  
We use statsmodeling's interface to look at the model coefficients as well as the p-values.  
We arbitrarily use a 5% p value cutoff and remove all variables that have a p value greater than 5%. 

Next we run another logisitic regression but only with the variables that were not dropped earlier.
We include the intercept although it had a pvalue issue.  
We find that there are still some variables with p value issues but we do not run another iteration. 

## Bayesian Analysis  

For the Bayesian analysis, we will use the pymc3 package. 
To start, we specify a model with non informative priors- the mean is 0 and the standard deviation is very high (10). 
By looking at the results from the regular logistic regression, we can see that all of the  
parameter estimates are within 1 standard deviation (all of the coefficients are < 10).  
We specify the logistic equation and specify that the dependent variable comes from a Bernoulli distribution. 

Next we use Maximum A Posteriori (MAP) to examine some of the parameter values. We use an optimized and alternate version. 
We can compare the MAP values with the results from the regular logistic regression. 
We see that most of the MAP values are pretty close to the regression results. 

Next we examine the trace -- the samples from the posterior distributions and get a better idea of the  
distributions for the coefficients of the Bayesian regression. Again the results converge to the estimates found in the 
regular logistic regression. 

More information about the Bayesian project can be found [here](https://github.com/malctaylor15/Wine_Quality/blob/master/Bayesian%20Start.ipynb)






