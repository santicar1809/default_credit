from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def all_models():
    seed=12345
    light_gbm_pipe=Pipeline([('scaler',StandardScaler()),
                            ('pca',PCA()),
                            ('lgbm',LGBMClassifier(random_state=seed))
                            ])
    
    light_gbm_params={
        'pca__n_components':[1,2],
        'lgbm__max_depth':[3,5,7],
        'lgbm__n_estimators':[10,50,100],
        'lgbm__learning_rate':[0.1,0.01,0.001],
        'lgbm__min_data_leaf':[20,40,60],
        'lgbm__lambda_l1':[0,0.5,1.0]
    }
    
    lgbm=['lgbm',light_gbm_pipe,light_gbm_params]
    
    xg_pipeline=Pipeline([('scaler',StandardScaler()),
                          ('pca',PCA()),
                          ('xg',XGBClassifier(random_state=seed))])
    
    xg_params={'pca__n_components':[1,2],
               'xg__max_depth':[3,5,7],
               'xg__n_estimators':[30,50,100],
               'xg__gamma':[0,0.1,0.3],
               'xg__alpha':[0,0.5,1.0]}
    
    xg=['xg',xg_pipeline,xg_params]
    
    cat_pipeline=Pipeline([('scaler',StandardScaler()),
                          ('pca',PCA()),
                          ('cat',CatBoostClassifier(random_state=seed))])
    
    cat_params={'pca__n_components':[1,2],
                'cat__n_estimators':[30,50,100],
                'cat__max_depth':[3,5,7],
                'cat__min_data_in_leaf':[20,40,60],
                'cat__l2_leaf_reg':[1,3,5]}
    
    cat=['cat',cat_pipeline,cat_params]
    
    rf_pipeline=Pipeline([('scaler',StandardScaler()),
                          ('pca',PCA()),
                          ('rf',RandomForestClassifier(random_state=seed))])
    
    rf_params={'pca__n_components':[1,2],
               'rf__n_estimators':[30,50,100],
               'rf__max_depth':[3,5,7],
               'rf__min_samples_leaf':[10,20,30]}
    
    rf=['rf',rf_pipeline,rf_params]
    
    dt_pipeline=Pipeline([('scaler',StandardScaler()),
                          ('pca',PCA()),
                          ('dt',DecisionTreeClassifier(random_state=seed))])
    
    dt_params={'pca__n_components':[1,2],
               'dt__max_depth':[3,5,7],
               'dt__min_samples_leaf':[10,20,30]}
    
    dt=['dt',dt_pipeline,dt_params]
    
    lr_pipeline=Pipeline([('scaler',StandardScaler()),
                ('pca',PCA()),
                ('lr',LogisticRegression(random_state=seed))])
    
    lr_params={}
    lr=['lr',lr_pipeline,lr_params]
    
    models=[lgbm,xg,cat,rf,dt,lr]
    return models