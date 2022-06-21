import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        self.X = [1, 2, 3]
        self.Y = [1, 2, 3]
        self.W = tf.placeholder(tf.float32)
        self.cost = None
        self.W_history = []
        self.cost_history = []


    def create_model(self):
        tf.set_random_seed(777)
        hypothesis = self.X * self.W
        self.cost = tf.reduce_mean(tf.square(hypothesis - self.Y))


    def fit(self):
        sess = tf.Session()
        

        for i in range(-30, 50):
            curr_W = i * 0.1
            curr_cost = sess.run(self.cost, {self.W: curr_W})
            self.W_history.append(curr_W)
            self.cost_history.append(curr_cost)
   
    
    def eval(self):
        plt.plot(self.W_history, cost_history)
        plt.show()

if __name__=='__main__':
    tf.disable_v2_behavior()
    s = Solution()
    s.create_model()
    s.fit()
    s.eval()
    
