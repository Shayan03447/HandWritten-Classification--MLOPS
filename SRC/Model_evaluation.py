import os
import logging
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


log_dir='log'
os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger("Model_evaluation")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir, "Model_evaluation.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_processed_data(processed_path):
    """Load processed test data """
    try:
        X_test=np.load(os.path.join(processed_path, "X_test.npy"))
        y_test=np.load(os.path.join(processed_path, "y_test.npy"))
        logger.debug("Processed test data loaded successfully")
        return X_test, y_test
    except Exception as e:
        logger.error("Error while loading processed test data: %s", e)
        raise

def load_trained_model(model_path):
    """Load trained model from saved path"""
    try:
        model=load_model(model_path)
        logger.debug("Trained model loaded successfully from %s",model_path)
        return model
    except Exception as e:
        logger.error("Error while loading the model: %s",e)
        raise
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return prediction and metrics"""
    try:
        y_pred_prob=model.predict(X_test)
        y_pred=np.argmax(y_pred_prob, axis=1)

        acc=accuracy_score(y_test, y_pred)
        cm=confusion_matrix(y_test, y_pred)
        report=classification_report(y_test,y_pred,digits=4)
        logger.debug("Model evaluation completed successfully")
        return acc, cm, report
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def save_evaluation_results(acc, cm, report, results_path):
    """Save evaluation metrics to a text file"""
    try:
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, "evaluation_results.txt")

        with open(results_file, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        logger.debug("Evaluation results saved successfully at %s", results_file)
    except Exception as e:
        logger.error("Error saving evaluation results: %s", e)
        raise
def main():
    try:
        # Parameters 
        processed_path = "./data/processed"
        model_path = "./model/mnist_ann_model.h5"
        results_path = "./results"

        # Load data & model
        X_test, y_test = load_processed_data(processed_path)
        model = load_trained_model(model_path)

        # Evaluate model
        acc, cm, report = evaluate_model(model, X_test, y_test)

        # Save evaluation results
        save_evaluation_results(acc, cm, report, results_path)

        logger.info("Model evaluation pipeline completed successfully ✅")

    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
