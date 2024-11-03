from src.preprocesing.load_dataset import load_dataset
from src.preprocesing.preprocessing import preprocessing_data
from src.EDA.EDA import eda

def main():
    data=load_dataset()
    preprocessed_data=preprocessing_data(data)
    eda()
    
    return preprocessed_data
main()