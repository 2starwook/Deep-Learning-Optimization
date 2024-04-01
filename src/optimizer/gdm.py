import tensorflow as tf


class GradientDescentMomentum(tf.Module):
    def __init__(self, learning_rate=1e-3, momentum=0.7):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.changes = list()
        self.title = f"Gradient descent momentum optimizer: "\
            f"learning rate={self.learning_rate} / "\
            f"momentum ={self.momentum}"

    def apply_gradients(self, grads, vars):
        if len(self.changes) <= 1: # self.changes gets wrapped by ListWrapper
            self.changes = [0. for _ in range(len(vars))]
        
        # Update variables 
        for i, (grad, var) in enumerate(zip(grads, vars)):
            change = self.changes[i]
            curr_change = self.learning_rate*grad + self.momentum*change
            var.assign_sub(curr_change)
            self.changes[i] = curr_change
