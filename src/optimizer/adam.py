import tensorflow as tf
import numpy as np


class Adam(tf.Module):
    def __init__(self, lr=0.07, betas=(0.9, 0.999), eps=0.01):
        """
        lr (float, optional): learning rate (default 0.07)
        betas (List[float, float], optional): 
            coefficients used for computing running averages of gradient and its square
            (default: (0.9), (0.999))
        eps (float, optional): small positive value to avoid divide by zero error
            (default: 0.001)
        """
        self.lr = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.eps = eps
        self.title = f"Adam optimizer: learning rate={self.lr} / "\
                        f"beta_1={self.beta_1} / "\
                        f"beta_2={self.beta_2} / "\
                        f"epsilon={self.eps}"

    def apply_gradients(self, grads, vars):
        t = tf.Variable(0.0, dtype=tf.float32, name='t')
        t.assign(t + 1)
        for grad, var in zip(grads, vars):
            m = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='m')
            v = tf.Variable(np.zeros(var.get_shape(), dtype='float32'), name='v') # velocity
            alpha_t = self.lr*tf.sqrt(1. - self.beta_2**t)/(1. - self.beta_1**t)
            m.assign(self.beta_1*m + (1. - self.beta_1)*grad)
            v.assign(self.beta_2*v + (1. - self.beta_2)*grad**2)
            var.assign(var - alpha_t*m/(tf.sqrt(v) + self.eps))
