import logging
import os
from tensorflow.keras.datasets import mnist
import numpy as np

log_dir='log'
os.makedirs(log_dir, exist_ok=True)

# Configuration
logger=logging.getLogger('Data_ingestion')
logger.setLevel('DEBUG')

console_handlar= logging.StreamHandler()
console_handlar.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "Data_ingestion.log")
file_handler= logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handlar.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handlar)
logger.addHandler(file_handler)


def load_data() -> tuple:
    """Load mnist dataset"""
    try:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        logger.debug("MNIST DATA LOADED SUCCESSFULLY")
        return (X_train,y_train),(X_test,y_test)
    except Exception as e:
        logger.error('UNEXPECTED ERROR WHILE LOADING THE DATA: %s', e)
        raise

def save_data(X_train,y_train,X_test,y_test, data_path: str) -> None:
    """Save raw train and test data as numpy array"""
    try:
        raw_data_path=os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)

        np.save(os.path.join(raw_data_path, "X_train.npy"),X_train)
        np.save(os.path.join(raw_data_path, "y_train.npy"),y_train)
        np.save(os.path.join(raw_data_path, "X_test.npy"),X_test)
        np.save(os.path.join(raw_data_path, "y_test.npy"),y_test)

        logger.debug("Raw data saved successfully at %s",raw_data_path)
    except Exception as e:
        logger.error("Error while saving raw data: %s ",e)
        raise

def main():
    try:
        (X_train,y_train),(X_test,y_test)=load_data()
        save_data(X_train,y_train,X_test,y_test,data_path="./Data")
    except Exception as e:
        logger.error('Data ingestion pipeline failed: %s',e)
        print(f"Error: {e}")

if __name__=="__main__":
    main()


