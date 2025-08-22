import os
import logging
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

log_dir='log'
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger("Model_building")
logger.setLevel("DEBUG")

console_handler= logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir, "Model_building.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
        processed_path="./Data/processed"
        save_path="./model"
        input_shape=(28,28)
        num_classes= 10
        epochs=10
        validation_split=0.2
        # Load data
        X_train, y_train, X_test, y_test=load_preprocess_data(processed_path=processed_path)

        # Build ANN model
        model=build_model(input_shape=input_shape,num_classes=num_classes) 

        # Train ANN model
        model, history=train_model(model, X_train,y_train,X_test,y_test, epochs=epochs, validation_split=validation_split)

        # Save train model
        save_model(model, save_path=save_path)
        logger.debug("Model building and training pipeline completted successfully")
    except Exception as e:
        logger.error("Model building pipeline failed: %s", e)
        print(f"Error: {e}")
if __name__=="__main__":
    main()


       