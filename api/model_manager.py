from pathlib import Path
import pickle
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str = "models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.model = None
        self.metadata = {}

    def load_model(self):
        # load latest model
        model_file = self.model_path / "model.pkl"
        metadata_file = self.model_path / "metadata.json"

        if not model_file.exists():
            raise FileNotFoundError("No model found")
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": "unknown",
                "accuracy": 0.0,
                "trained_at": "unknown"
            }
        
        logger.info(f"Loaded model version {self.metadata["version"]}")
    
    def predict(self, X: np.ndarray):
        # 0 = down, 1 = up
        if self.model == None:
            raise RuntimeError("Model is not loaded")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        # get the prediction probabilities
        if self.model == None:
            raise RuntimeError("Model is not loaded")
        return self.model.predict_proba(X)
    
    def get_version(self):
        return self.metadata.get("version", "unknown")

    def get_accuracy(self):
        return self.metadata.get("accuracy", 0.0)
    
    def get_trained_date(self):
        return self.metadata.get("trained_at", "unknown")
