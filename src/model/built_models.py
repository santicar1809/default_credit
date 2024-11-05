from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.model.hyperparameters import all_models
import joblib
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

def evaluate_model(data,model):
    best_estimator=model.best_estimator_
    pred=best_estimator.predict(data[2])
    proba=best_estimator.predict_proba(data[2])[:,1]
    accuracy=accuracy_score(data[3],pred)
    f1_val=f1_score(data[3],pred)
    auc_score=roc_auc_score(data[3],proba)
    return best_estimator,accuracy,f1_val,auc_score
    

def model_data(data):
    models=all_models()
    models_path='./files/output/output-fit/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    reports_path='./files/output/reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    scores=[]
    for model in models:
        
        grid=GridSearchCV(model[1],model[2],n_jobs=-1,scoring='roc_auc')
        grid.fit(data[0],data[1])
        best_estimator,accuracy,f1_val,auc_score=evaluate_model(data,grid)
        scores.append([model[0],best_estimator,accuracy,f1_val,auc_score])
        joblib.dump(best_estimator,models_path+f'{model[0]}.joblib')

    results=pd.DataFresults=pd.DataFrame(scores,columns=['model','best_estimator','accuracy','f1','roc_auc'])
    tf_results=ANN(data)
    results_df=pd.concat([results,tf_results])
    results_df.to_csv(reports_path+'results.csv',index=False)
    return results_df

def ANN(data):
    models_path='./files/output/output-fit/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    model=Sequential([
        Dense(128,activation='relu',input_shape=(data[0].shape[1],),kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64,activation='relu',input_shape=(data[0].shape[1],),kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32,activation='relu',kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(16,activation='relu',kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1,activation='sigmoid')
        ])
    model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC()])
    model.summary()
    early_stoping=EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
    
    history=model.fit(data[0],data[1],epochs=200,batch_size=32,validation_data=(data[2],data[3]),callbacks=[early_stoping])

    # Evaluation
    y_pred_proba=model.predict(data[2]).flatten()
    y_pred=(y_pred_proba>0.5).astype('int')
    accuracy_val=accuracy_score(data[3],y_pred)
    f1_val=f1_score(data[3],y_pred)
    auc_score=roc_auc_score(data[3],y_pred_proba)
    scores=['keras',accuracy_val,f1_val,auc_score]
    scores_df=pd.DataFrame({'model':[scores[0]],'accuracy':[scores[1]],'f1':[scores[2]],'roc_auc':[scores[3]]})
    model.save(models_path+'neural_network.h5')
    return scores_df
    