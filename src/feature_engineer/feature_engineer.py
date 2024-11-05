import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import os

test_path='./test/dataset/'

if not os.path.exists(test_path):
    os.makedirs(test_path)

def up_sampling(features,target,repeat):
    seed=12345    
    features_zeros=features[target==0]
    features_ones=features[target==1]
    target_zeros=features[target==0]
    target_ones=features[target==1]
    
    features_upsampled=pd.concat([features_zeros] + [features_ones*repeat])
    target_upsampled=pd.concat([target_zeros] + [target_ones*repeat])
    
    features_upsampled,target_upsampled=shuffle(features_upsampled,target_upsampled,random_state=seed)
    return features_upsampled,target_upsampled

def downsampling(features,target,fraction):
    seed=12345
    features_zeros=features[target==0]
    features_ones=features[target==1]
    target_zeros=features[target==0]
    target_ones=features[target==1]
    
    features_downsampled=pd.concat([features_ones]+[features_zeros.sample(frac=fraction)])
    target_downsampled=pd.concat([target_ones]+[target_zeros.sample(frac=fraction)])
    features_downsampled,target_downsampled=shuffle(features_downsampled,target_downsampled,random_state=seed)
    return features_downsampled,target_downsampled    

def feature_engineering(data):
    seed=12345
    df_train,df_test=train_test_split(data,test_size=0.3,random_state=seed)
    features=df_train.drop(['default_payment_next_month'],axis=1)
    target=df_train['default_payment_next_month']
    # Balanceo
    balancer=SMOTE(random_state=seed)
    features,target=balancer.fit_resample(features,target)
    target.value_counts()
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.30,random_state=seed)
    df_test.to_csv(test_path+'df_test.csv',index=False)
    
    return features_train,target_train,features_valid,target_valid



