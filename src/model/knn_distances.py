import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from src.preprocesing.load_dataset import load_dataset
from src.preprocesing.preprocessing import preprocessing_data
from src.feature_engineer.feature_engineer import feature_engineering
from collections import Counter
data=feature_engineering(preprocessing_data(load_dataset()))
features_train=np.array(data[0])
target_train=np.array(data[1])
features_valid=np.array(data[2])
target_valid=np.array(data[3])



class KnnClassifier:
    
    def __init__(self,k=3):
        self.k=k
        
    def fit(self,features,target):
        self.features=np.array(features)
        self.target=np.array(target)
    
    def predict(self,features_valid):
        predicted_labels=[self._predict(x) for x in np.array(features_valid)]
        return np.array(predicted_labels)
            

    def _predict(self,x):
        # Compute distances
        distances=[distance.euclidean(x,x_train) for x_train in self.features]
        # get k nearest samples, labels
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.target[i] for i in k_indices]
        # majority vote, most common label
        most_common_equals=Counter(k_nearest_labels).most_common(1)
        return most_common_equals[0][0]
    
knn=KnnClassifier()

knn.fit(features_train,target_train)
pred=knn.predict(features_valid)
print(pred)
accuracy_val=accuracy_score(target_valid,pred)
print(f'This is the accuracy of your KNN model: {accuracy_val:.2f}')


x_axis = np.array([0., 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])

y_axis  = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])

deliveries = np.array([5, 7, 4, 3, 5, 2, 1, 1])

town = [
    'Willowford',
    'Otter Creek',
    'Springfield',
    'Arlingport',
    'Spadewood',
    'Goldvale',
    'Bison Flats',
    'Bison Hills',
]

data=pd.DataFrame({'x_axis':x_axis,'y_axis':y_axis,'deliveries':deliveries},index=town)

vectors=data[['x_axis','y_axis']].values

distances=[]
dist=[]

for i in range(len(town)):
    distances=[]
    for j in range(len(town)):
        distan=distance.euclidean(vectors[i],vectors[j])
        distances.append(distan)
    dist.append(distances)

deliveries_in_week=[]
for shop in dist:
    total_distance=sum(shop*deliveries)
    deliveries_in_week.append(total_distance)
    
deliveries_in_week_df=pd.DataFrame({'deliveries_in_week':deliveries_in_week},index=town)

min_distance=deliveries_in_week_df.idxmin()



class Camiseta:
    
    def __init__(self,marca,precio,talla,color):
        self.marca=marca
        self.precio=precio
        self.talla=talla
        self.color=color
        self.rebajada=False
    
    def discount(self,porcentaje):
        self.precio -=self.precio*porcentaje/100
        if porcentaje < 100:
            self.rebajada=True
        
    def infoProducto(self):
        info=f"Descripcion de la camiseta:\nMarca: {self.marca}\Precio: {self.marca}"
    
camiseta=Camiseta("Nike",100,"S","lila")
camiseta.discount(30)
print(camiseta.precio)