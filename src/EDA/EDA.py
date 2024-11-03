import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import os



def eda():
    rep_path='./files/output/reports/'
    if not os.path.exists(rep_path):
        os.makedirs(rep_path)
    eda_path='./files/output/figs/'
    if not os.path.exists(eda_path):
        os.makedirs(eda_path)

    data=pd.read_csv('./files/intermediate/intermediate_df.csv')

    describe_df=data.describe()
    info_df=data.info()

    with open(rep_path+'describe.csv','w') as f:
        f.write(describe_df.to_string())


    #with open(eda_path+'info.csv','w') as f:
    #    f.write(info_df)
    # Education vs balance
    fig,ax=plt.subplots()
    sns.barplot(x='education',y='limit_bal',hue='sex',data=data,ax=ax)
    ax.set_title('Balance per education')
    fig.savefig(eda_path+'education.png')

    #Scatter plot
    fig1=px.scatter(data,x='total_bill',y='total_pay',title='total_bill-vs-total_pay')
    fig1.write_image(eda_path+'total_bill-vs-total_pay.png',format='png',scale=2)

    #Age influence
    fig2,ax2=plt.subplots(3,1,figsize=(10,10))
    sns.lineplot(x='age',y='limit_bal',hue='sex',data=data,ax=ax2[0])
    sns.lineplot(x='age',y='total_pay',hue='sex',data=data,ax=ax2[1])
    sns.lineplot(x='age',y='total_bill',hue='sex',data=data,ax=ax2[2])
    ax2.set_title('Age vs metrics')
    plt.tight_layout
    fig2.savefig(eda_path+'age.png')

    # Repayment_status
    data_reps=data.groupby(['repayment_status'])['repayment_status'].count()
    fig3,ax3=plt.subplots()
    ax3.bar(data_reps.index,data_reps)
    ax3.set_title('Repayment_status')
    fig3.savefig(eda_path+'repayment_status.png')

    # Total_bill marriage
    data_long = pd.melt(data, id_vars='marriage', value_vars=['limit_bal', 'total_bill', 'total_pay'],
                        var_name='metric', value_name='value')
    fig4,ax4=plt.subplots()
    sns.barplot(x='marriage',y='value',hue='metric',data=data_long,ax=ax4)
    ax4.set_title('Marriage vs bills')
    fig4.savefig(eda_path+'marriage.png')

    # Clases distribution
    clases_df=data.groupby(['default_payment_next_month'])['default_payment_next_month'].count()
    fig5,ax5=plt.subplots()
    ax5.bar(clases_df.index,clases_df)
    ax5.set_title('Clases distribution')
    fig5.savefig(eda_path+'clases.png')