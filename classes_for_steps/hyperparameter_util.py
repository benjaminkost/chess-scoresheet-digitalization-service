class ModelHyperparameter:
    """The base class for hyperparameters."""
    def save_hyperparameter(self, ignore=[]):
        raise NotImplemented

class TrOCRHyperparameter(ModelHyperparameter):
    def __init__(self):
        self.save_hyperparameter(ignore=[])
