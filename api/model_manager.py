from pathlib import Path

class ModelManager:
    def __init__(self, model_path: str = "models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.model = None
        self.metadata = {}