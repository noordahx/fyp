# utility functions (early stopping classes, metics, etc.)

class EarlyStopPatience:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
    
    def __call__(self, current_metric):
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if current_metric <= self.best_metric:
            self.counter += 1
        else:
            self.best_metric = current_metric
            self.counter = 0
        
        if self.counter >= self.patience:
            return True
        return False