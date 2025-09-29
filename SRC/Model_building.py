import os
import logging
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import yaml
import mlflow
import mlflow.keras


log_dir='log'
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger("Model_building")
logger.setLevel("DEBUG")

console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "Model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameter from the yaml file"""
    try:
        with open(params_path, 'r') as file:
            params=yaml.safe_load(file)
        logger.debug("parameters retrived from :%s", params_path)
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

def load_preprocess_data(processed_path):
    """Load processed (normalized) numpy array"""
    try:
        X_train=np.load(os.path.join(processed_path, "X_train.npy"))
        y_train=np.load(os.path.join(processed_path, "y_train.npy"))
        X_test=np.load(os.path.join(processed_path, "X_test.npy"))
        y_test=np.load(os.path.join(processed_path, "y_test.npy"))
        logger.debug("Processed data loaded successfully")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error("Error while loading processed data : %s", e)
        raise

def build_model(input_shape, num_classes):
    """Build ANN model with sparse_categorical_crossentropy"""
    try:
        model=Sequential([
            Flatten(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        logger.debug("Model build and compiled successfully")
        return model
    except Exception as e:
        logger.error("Error while building the model : %s", e)
        raise

def train_model(model, X_train, y_train, X_test, y_test, epochs, validation_split):
    """Train ANN model"""
    try:
        history=model.fit(
            X_train,y_train,
            validation_split=validation_split,
            validation_data=(X_test, y_test),
            epochs=epochs,
            verbose=1
        )
        logger.debug("Model training completed")
        return model, history
    except Exception as e:
        logger.error("Error during training: %s", e)
        raise

def save_model(model, save_path):
    """Saved trained model"""
    try:
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, "mnist_ann_model.h5"))
        logger.debug("Model saved successfully at %s",save_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise

def main():
    try:
        params=load_params(params_path='params.yaml')
        processed_path=params["model_building"]["process_data_path"]
        save_path=params["model_building"]["model_save_path"]
        input_shape=tuple(params["model_building"]["input_shape"])
        num_classes= params["model_building"]["num_classes"]
        epochs=params["model_building"]["epochs"]
        validation_split=params["model_building"]["validation_split"]

        # Mlflow experiment tracking
        with mlflow.start_run():
            # log parameters
            mlflow.log_params({
                "input_shape":input_shape,
                "num_classes":num_classes,
                "epochs":epochs,
                "validation_split":validation_split
            })

            # Load data
            X_train, y_train, X_test, y_test=load_preprocess_data(processed_path=processed_path)

            # Build ANN model
            model=build_model(input_shape=input_shape,num_classes=num_classes) 

            # Train ANN model
            model, history=train_model(model, X_train,y_train,X_test,y_test, epochs=epochs, validation_split=validation_split)

            # Log metrics
            final_acc=history.history["accuracy"][-1]
            val_acc=history.history["val_accuracy"][-1]
            mlflow.log_metric("train_accuracy", final_acc)
            mlflow.log_metric("val_accuracy",val_acc)



            # Save and log train model
            save_model(model, save_path=save_path)
            mlflow.keras.log_model(model,"model")
            logger.debug("Model building and training pipeline completted successfully")
    except Exception as e:
        logger.error("Model building pipeline failed: %s", e)
        print(f"Error: {e}")
if __name__=="__main__":
    main()


       