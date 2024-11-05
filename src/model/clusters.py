from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
import pandas as pd
import numpy as np
from sklearn.neighbors import  NearestNeighbors
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import joblib


def clustering_data(data):
    eda_path='./files/output/figs/'
    output_path = './files/output/output-fit/'
    # Kmeans
    # Elbow method
    seed=12345
    features=data[['limit_bal','total_bill','total_pay']]
    wcss=[]
    for i in range(1,10):
        kmeans=KMeans(n_clusters=i,random_state=seed)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(range(1,10),wcss,marker='o',linestyle='--')
    ax.set_title('Numero de clusteres')
    ax.set_xlabel('Numero de clusteres')
    ax.set_ylabel('WCSS')
    fig.show()
    fig.savefig(eda_path+'elbow_method.png')

    # Aplicamos Kmeans con 6 clusteres

    kmeans=KMeans(n_clusters=6,random_state=seed)
    kmeans.fit(features)
    joblib.dump(kmeans,output_path+'kmeans_elbow.joblib')
    centroids=pd.DataFrame(kmeans.cluster_centers_,columns=features.columns)
    for i in centroids.columns:
        centroids[i]=centroids[i].astype('float')
    features['cluster']=kmeans.labels_.astype('str')
    centroids['cluster']=['0 centroid','1 centroid','2 centroid','3 centroid','4 centroid','5 centroid']

    fig1,ax1=plt.subplots(3,1,figsize=(10,10))
    sns.scatterplot(x='limit_bal',y='total_bill',hue='cluster',data=features,ax=ax1[0])
    ax1[0].scatter(centroids['limit_bal'],centroids['total_bill'],marker='*',s=200,c='red')

    sns.scatterplot(x='limit_bal',y='total_pay',hue='cluster',data=features,ax=ax1[1])
    ax1[1].scatter(centroids['limit_bal'],centroids['total_pay'],marker='*',s=200,c='red')

    sns.scatterplot(x='total_bill',y='total_pay',hue='cluster',data=features,ax=ax1[2])
    ax1[2].scatter(centroids['total_bill'],centroids['total_pay'],marker='*',s=200,c='red')

    plt.tight_layout()
    fig1.savefig(eda_path+'kmeans_clusters.png')

    # Silhouette method
    sil_list=[]
    cluster_list=np.arange(2,13)
    for i in cluster_list:
        kmeans=KMeans(n_clusters=i,random_state=seed)
        pred=kmeans.fit_predict(features)
        score=silhouette_score(features,pred)
        sil_list.append(score)
    fig2,ax2=plt.subplots(figsize=(10,6))
    ax2.plot(cluster_list,sil_list,marker='o',linestyle='--')
    ax2.set_title('Silhouette method')
    fig2.savefig(eda_path+'silouette_method.png')

    # We have the same number of clusters with the silhouette method

    #DBSCAN

    knn=NearestNeighbors(n_neighbors=7)
    model=knn.fit(features)
    distances,indices= knn.kneighbors(features)
    distances=np.sort(distances,axis=0)
    distances=distances[:,1]
    fig3,ax3=plt.subplots(figsize=(10,6))
    ax3.grid()
    ax3.plot(distances)
    ax3.set_title('KNN-neighbors')
    fig3.savefig(eda_path+'knn.png')

    dbscan=DBSCAN(eps=0.02,min_samples=8)
    features=features.values
    labels=dbscan.fit_predict(features)
    joblib.dump(dbscan,output_path+'dbscan.joblib')
    # Extract core samples (points that are part of a dense cluster)
    core_samples_mask=np.zeros_like(labels,dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_]=True

    # Number of clusters (ignoring noise if present)
    n_clusters=len(set(labels))-(1 if -1 in labels else 0)

    # Plotting cluster
    unique_labels=set(labels)
    colors=[plt.cm.Spectral(each) for each in np.linspace(0,1,len(unique_labels))]

    fig4,ax4=plt.subplots(figsize=(10,10))
    for label, color in zip(unique_labels,colors):
        if label ==-1:
            #Black for outliers
            color=[0,0,0,1]

            class_member_mask = (labels==label)

            xy=features[class_member_mask & core_samples_mask]
            plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor=tuple(color),markeredgecolor='k',markersize=10)

            xy = features[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

        ax4.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
        ax4.set_xlabel("limit_bal")
        ax4.set_ylabel("total_bill")
        plt.tight_layout()
        fig4.show()
        fig4.savefig(eda_path+'dbscan_clusters_1.png')

    fig5,ax5=plt.subplots(figsize=(10,10))
    for label, color in zip(unique_labels,colors):
        if label ==-1:
            #Black for outliers
            color=[0,0,0,1]

            class_member_mask = (labels==label)

            xy=features[class_member_mask & core_samples_mask]
            plt.plot(xy[:,0],xy[:,2],'o',markerfacecolor=tuple(color),markeredgecolor='k',markersize=10)

            xy = features[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 2], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

        ax5.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
        ax5.set_xlabel("limit_bal")
        ax5.set_ylabel("total_pay")
        plt.tight_layout()
        fig5.show()
        fig5.savefig(eda_path+'dbscan_clusters_2.png')

    fig6,ax6=plt.subplots(figsize=(10,10))
    for label, color in zip(unique_labels,colors):
        if label ==-1:
            #Black for outliers
            color=[0,0,0,1]

            class_member_mask = (labels==label)

            xy=features[class_member_mask & core_samples_mask]
            plt.plot(xy[:,1],xy[:,2],'o',markerfacecolor=tuple(color),markeredgecolor='k',markersize=10)

            xy = features[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

        ax6.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
        ax6.set_xlabel("limit_bal")
        ax6.set_ylabel("total_pay")
        plt.tight_layout()
        fig6.show()
        fig6.savefig(eda_path+'dbscan_clusters_2.png')

    # Hierachical clustering
    fig7,ax7=plt.subplots(figsize=(10,10))
    linkage_data=linkage(features,method='ward',metric='euclidean')
    dendrogram(linkage_data)
    ax7.set_title('Dendogram')
    plt.tight_layout()
    fig7.savefig(eda_path+'dendogram.png')
    fig7.show()

    hierac=AgglomerativeClustering(n_clusters=6,linkage='ward')
    hierac.fit(features)
    joblib.dump(hierac,output_path+'hierac.joblib')
    features=data[['limit_bal','total_bill','total_pay']]
    features['cluster']=hierac.labels_
    fig8=plt.figure()
    ax8=fig8.add_subplot(111,projection='3d')
    x=np.array(features['limit_bal'])
    y=np.array(features['total_bill'])
    z=np.array(features['total_pay'])
    c=features['cluster']
    sc=ax8.scatter(x,y,z,c=c,cmap='viridis',s=50)
    plt.title('Hierarchical clusters')
    ax8.set_xlabel('Limit balance')
    ax8.set_ylabel('Total bill')
    ax8.set_zlabel('Total pay')
    plt.tight_layout()
    fig8.savefig(eda_path+'hierarc.png')
