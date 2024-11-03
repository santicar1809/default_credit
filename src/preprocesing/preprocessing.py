import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import sys
import os
from sklearn.ensemble import IsolationForest



def snake_case(col):
    s1=re.sub('(.)([A-Z][a-z]+)',r'\1_\2',col)
    s2=re.sub('([a-z])([A-Z])',r'\1_\2',s1)
    return s1.replace(' ','_').lower()

def column_snake(data):
    new_cols=[]
    for col in data.columns:
        col_edited=snake_case(col)
        new_cols.append(col_edited)
    data.columns=new_cols
    return data


def preprocessing_data(data):
    #Reemplazamos las columnas en formato snake case
    data=column_snake(data)

    if data['marriage'].isna().sum() > 1:
        data.dropna(inplace=True)
    if data.duplicated().sum() > 1:
        data.drop_duplicates(inplace=True)

    eda_path='./files/output/figs/'

    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    for column in data.columns:
        fig,ax=plt.subplots()
        ax.boxplot(data[column])
        ax.set_title(f'{column} box plot')
        fig.savefig(eda_path+f'{column}_boxplot.png')

    iso=IsolationForest(n_estimators=100)
    estimator=iso.fit_predict(data)
    estimator

    data['outlier']=estimator

    data[data['outlier']==-1]['id'].count()/data.shape[0]

    # Since we have 6% of the outliers, we're gonna drop them so it doesn't biase our data.
    data = data[data['outlier']!=-1]

    data['total_bill']=data.loc[:,'bill_amt1':'bill_amt6'].sum(axis=1)
    data['total_pay']=data.loc[:,'pay_amt1':'pay_amt6'].sum(axis=1)
    data['repayment_status']=data['pay_6']
    data_excluded=data.loc[:,'pay_0':'pay_amt6']
    data.drop(data_excluded.columns,axis=1,inplace=True)
    data.drop(['outlier'],axis=1)
    
    output_data_path='./files/intermediate/'
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    data.to_csv(output_data_path+'intermediate_df.csv',index=False)
    return data