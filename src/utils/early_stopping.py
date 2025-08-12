class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, current_value: float) -> bool:
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop