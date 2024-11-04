from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.model.hyperparameters import all_models
import joblib
import os
import pandas as pd
from src.preprocesing.load_dataset import load_dataset
from src.preprocesing.preprocessing import preprocessing_data
from src.feature_engineer.feature_engineer import feature_engineering
data=feature_engineering(preprocessing_data(load_dataset()))

def evaluate_model(data,model):
    best_estimator=model.best_estimator_
    pred=best_estimator.predict(data[2])
    accuracy=accuracy_score(data[3],pred)
    f1_val=f1_score(data[3],pred)
    auc_score=roc_auc_score(data[3],pred)
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
    results.to_csv(reports_path+'results.csv',index=False)
    return results



