class OptimizerSettings:
    def __init__(self, lr=5e-4, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def display_params(self):
        print('lr: ', self.lr)
        print('weight_decay: ', self.weight_decay)
