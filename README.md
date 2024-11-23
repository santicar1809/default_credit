# Description

This project consist in using information of Taiwanese credit card customers such as gender, education, payment history, age, repayment status and so on, to predict the defaulted customers of the credit card branch.

First we made the preprocessing of the data, checking missing values, duplicated values, anomalies, outliers and so on with machine learning algorithms such as Isolation Forest.

After thar we performed an EDA to identify insights and key information about the customers like what kind of customers are being defaulted, the influence of the age, marital status, or payment history.

Additionaly, we apply feature engineeer to balance the labels of the defaulted customers in order to prevent bias and we split data in the train dataset, the validation dataset and the test dataset separating features and the target.

Done the engineering we train our models to predict defaulted customers using a pipeline that scalates data, decompose data to use only adecuate features for the algorithms and trains the model.

Besides the supervised algorithms, we used non supervised algorithms to cluster the different kind of customers based on 3 features, total bill, total pay and balance limit.

At last, we tested data with our models previously trained with our testing dataset and save the results to define the best model.

# Dataset information

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005