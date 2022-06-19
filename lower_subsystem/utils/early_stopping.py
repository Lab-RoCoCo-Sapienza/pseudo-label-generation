class EarlyStopping:
    """
    Monitor a metric and stop training when it stops improving.

    Args:
        mode (str): one of 'minimize' or 'maximize' (default). In 'minimize' mode, training will stop when the
                    quantity monitored has stopped decreasing and in 'maximize' mode it will stop when the quantity
                    monitored has stopped increasing.
        patience (int): number of epochs to wait for improvement before terminating.
                        The counter resets after each improvement.

        min_delta (float > 0.0): minimum change in the monitored value to be considered an improvement.
    """
    def __init__(self, mode='maximize', patience=5, min_delta=0.0):
        assert mode in ['maximize', 'minimize']
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0
        self.wait_count = 0
        self.has_improved = True
        if mode == 'maximize':
            self.best_value = float('-inf')
        else:
            self.best_value = float('inf')

    def on_epoch_end(self, value, epoch):
        if (
            (self.mode == 'maximize' and (value - self.best_value) > self.min_delta)
            or (self.mode == 'minimize' and (self.best_value - value) > self.min_delta)
        ):
            self.best_value = value
            self.best_epoch = epoch
            self.wait_count = 0
            self.has_improved = True
        else:
            self.has_improved = False
            if self.patience > 0:
                self.wait_count += 1
                print(f"No improvement -> patience: {self.wait_count}/{self.patience}")

    def should_stop(self):
        if self.patience > 0:
            if self.wait_count == self.patience:
                return True
            else:
                return False
