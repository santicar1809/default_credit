import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def features_2(data):
    numeric=['limit_bal','total_bill','total_pay']
    scaler=MinMaxScaler()
    data_scaled=scaler.fit_transform(data[numeric])
    data_scaled=pd.DataFrame(data_scaled,columns=numeric,index=data.index)
    data[numeric]=data_scaled
    return data