import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score
from tensorflow.keras.models import load_model
import joblib

def test_data():
    data =pd.read_csv('./test/dataset/df_test.csv')

    features=data.drop(['default_payment_next_month'],axis=1)

    target=data['default_payment_next_month']

    models_name=['cat','dt','lgbm','lr','rf','xg']

    results=[]
    for i in models_name:
        model=joblib.load(f'./files/output/output-fit/{i}.joblib')
        pred=model.predict(features)
        proba=model.predict_proba(features)[:,1]
        accuracy_val=accuracy_score(target,pred)
        f1_val=f1_score(target,pred)
        roc_auc_val=roc_auc_score(target,proba)
        results.append([i,accuracy_val,f1_val,roc_auc_val])

    results_df=pd.DataFrame(results,columns=['model_name','accuracy','f1','roc_auc'])

    neural_model=load_model('./files/output/output-fit/neural_network.h5')

    prob=neural_model.predict(features).flatten()
    pred=(prob>0.5).astype('int')
    prob=neural_model.predict(features).ravel()
    accuracy_val=accuracy_score(target,pred)
    f1_val=f1_score(target,pred)
    roc_auc_val=roc_auc_score(target,prob)
    results_nn=['Keras',accuracy_val,f1_val,roc_auc_val]
    results_nn_df=pd.DataFrame({'model_name':[results_nn[0]],'accuracy':[results_nn[1]],'f1':[results_nn[2]],'roc_auc':[results_nn[3]]})

    results_test=pd.concat([results_df,results_nn_df])
    results_test.to_csv('./test/dataset/results_test.csv',index=False)
    return results_test

results=test_data()