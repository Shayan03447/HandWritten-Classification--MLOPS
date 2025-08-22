import logging
import os
import numpy as np
log_dir='log'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir, "Data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def normalize_data(X):
    """Normalize pixel value to the range of [0,1]"""
    try:
        X=X.astype("float32")/255.0
        logger.debug("DATA NORMALIZED TO [0,1]")
        return X
    except Exception as e:
        logger.error("ERROR DURING NORMALIZATION %s", e)
        raise

def save_normalized(X_train,y_train,X_test,y_test,save_path="./data/processed"):
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
        raw_path="./Data/raw"
        X_train, y_train, X_test, y_test=load_raw_data(raw_path)

        X_train=normalize_data(X_train)
        X_test=normalize_data(X_test)

        save_normalized(X_train, y_train, X_test,y_test)
        logger.debug("Data preprocessing complete")
    except Exception as e:
        logger.error("Data preprocessing pipeline failed: %s", e)
        print(f"Error: {e}")
    
if __name__=="__main__":
    main()




