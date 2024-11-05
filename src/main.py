from src.preprocesing.load_dataset import load_dataset
from src.preprocesing.preprocessing import preprocessing_data
from src.EDA.EDA import eda
from src.feature_engineer.feature_engineer import feature_engineering
from src.feature_engineer.feature_engineer_2 import features_2
from src.model.clusters import clustering_data
from src.model.built_models import model_data

def main():
    data=load_dataset()
    preprocessed_data=preprocessing_data(data)
    eda()
    processed_data=feature_engineering(preprocessed_data)
    results=model_data(processed_data)
    clustered_data=features_2(preprocessed_data)
    clustering_data(clustered_data)
    return results
main()