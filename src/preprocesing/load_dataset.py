import pandas as pd
import re
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

def load_dataset():
    df=pd.read_excel('./files/default_credit_card.xlsx',sheet_name='Data',header=1)
    return df




#df=pd.read_csv('./files/default_credit_card.csv',sep=";",header=1)
#df=pd.read_excel('./files/default_credit_card.xlsx',sheet_name='Data',header=1)
#df.info()
#df.describe()
#
##Preprocessing
#
#df.isna().sum()
#df.duplicated().sum()
#
#def snake_case(col):
#    s1=re.sub('(.)([A-Z][a-z]+)',r'\1_\2',col)
#    s2=re.sub('([a-z0-9])([A-Z])',r'\1_\2',s1)
#    return s2.replace(' ', '_').lower()
#
#def column_snake(data):
#    new_col=[]
#    for column in data.columns:
#        col_ed=snake_case(column)
#        new_col.append(col_ed)
#    data.columns=new_col
#    return data
#
#df=column_snake(df)   
#df
##EDA
#list=[57,29,32]
#df.query("age in @list")[['id','default_payment_next_month']]
##df['sex'].where(df['sex']!=1,'Male',inplace=True)
#
##matplotlib
##plt.clf()
#fig,ax=plt.subplots()
#ax.plot(df['age'],df['bill_amt4'],'o',color='orange')
##plt.show()
#
##plt.clf()
##fig1,ax1=plt.subplots()
##ax1.bar(df['sex'],df['bill_amt4'],color='black')
##plt.show()
#
##plt.clf()
#fig2,ax2=plt.subplots()
#ax2.boxplot(df['bill_amt4'])
##plt.show()
#
##Seaborn
##plt.clf()
#fig,ax=plt.subplots(2,1,figsize=(20,20))
#sns.scatterplot(x='bill_amt4',y='pay_amt1',hue='sex',data=df,ax=ax[0])
#ax[0].scatter(df['bill_amt4'],df['pay_amt1'],marker='*',c='red')
#
#sns.lineplot(x='age',y='bill_amt4',hue='marriage',data=df,ax=ax[1])
#ax[1].plot(df['age'],df['bill_amt4'])
#
#plt.tight_layout()
##plt.show()
#
## Plotly
#
#fig=px.histogram(df,x='sex',y='age',color='marriage',barmode='group')
##fig.show()
#
##Correlation
#
#data_cor=df.corr()
#fig,ax=plt.subplots(figsize=(20,20))
#sns.heatmap(data_cor,annot=True,cmap='coolwarm',fmt=".2f",ax=ax)
#plt.show()