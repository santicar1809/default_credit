from src.preprocesing.load_dataset import load_dataset
from src.preprocesing.preprocessing import preprocessing_data
from src.EDA.EDA import eda
from src.feature_engineer.feature_engineer import feature_engineering

def main():
    data=load_dataset()
    preprocessed_data=preprocessing_data(data)
    eda()
    processed_data=feature_engineering(preprocessed_data)
    return processed_data[0]
main()