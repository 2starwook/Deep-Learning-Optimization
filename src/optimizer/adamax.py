import tensorflow as tf
import numpy as np


class Adamax(tf.Module):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        lr (float, optional): learning rate (default 0.001)
        betas (List[float, float], optional): 
            exponential decay rate for 1st moment estimates and infinity norm
            (default: (0.9), (0.999))
        eps (float, optional): small positive value to avoid divide by zero error
            (default: 1e-8)
        """
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.eps = eps
        self.title = f"Adamax optimizer: learning rate={self.lr} / "\
                        f"beta_1={self.beta_1} / "\
                        f"beta_2={self.beta_2} / "\
                        f"epsilon={self.eps}"

    def apply_gradients(self, grads, vars):
        t = tf.Variable(0.0, dtype=tf.float32, name='t')
        t.assign(t+1)
        for grad, var in zip(grads, vars):
            m = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='m')
            u = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='u')
            m.assign(self.beta_1*m + (1. - self.beta_1)*grad)
            u.assign(tf.maximum(self.beta_2*u, tf.abs(grad)))
            var.assign(var - (self.lr/(1. - self.beta_1**t))*m/(u + self.eps))
