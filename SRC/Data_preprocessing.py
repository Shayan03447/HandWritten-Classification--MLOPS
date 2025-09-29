import logging
import os
import numpy as np
import yaml

log_dir='log'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "Data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from the yaml file """
    try:
        with open(params_path, 'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameter retrived from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
    

def load_raw_data(raw_path: str):
    """Load raw mnist numpy array"""
    try:
        X_train=np.load(os.path.join(raw_path, "X_train.npy"))
        y_train=np.load(os.path.join(raw_path, "y_train.npy"))
        X_test=np.load(os.path.join(raw_path, "X_test.npy"))
        y_test=np.load(os.path.join(raw_path, "y_test.npy"))
        logger.debug("RAW DATA LOADED SUCCESSFULLY")
        return X_train,y_train,X_test,y_test
    except Exception as e:
        logger.error("ERROR WHILE LOADING THE RAW DATA: %s", e)
        raise

def normalize_data(X, norm_factor=255.0):
    """Normalize pixel value to the range of [0,1]"""
    try:
        X=X.astype("float32")/norm_factor
        logger.debug("DATA NORMALIZED TO [0,1]")
        return X
    except Exception as e:
        logger.error("ERROR DURING NORMALIZATION %s", e)
        raise

def save_normalized(X_train,y_train,X_test,y_test,save_path="./Data/processed"):
    """Save normalized numpy array"""
    try:
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, "X_train"), X_train)
        np.save(os.path.join(save_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_path, "X_test.npy"), X_test)
        np.save(os.path.join(save_path, "y_test.npy"),y_test)
        logger.debug("NORMALIZED DATA SAVED SUCCESSFULLY")
    except Exception as e:
        logger.error("Error while saving normalized data: %s", e)
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        raw_path=params["data_preprocessing"]["raw_data_path"]
        save_path=params["data_preprocessing"]["processed_data_path"]
        norm_factor=params["data_preprocessing"]["normalization_factor"]

        X_train, y_train, X_test, y_test=load_raw_data(raw_path)

        X_train=normalize_data(X_train, norm_factor)
        X_test=normalize_data(X_test, norm_factor)

        save_normalized(X_train, y_train, X_test,y_test,save_path)
        logger.debug("Data preprocessing complete")
    except Exception as e:
        logger.error("Data preprocessing pipeline failed: %s", e)
        print(f"Error: {e}")
    
if __name__=="__main__":
    main()




