import numpy as np
import tensorflow as tf

SMALL = 0.0000001

# ===========================
# ======= U T I L S =========
# ===========================

# this function is 'borrowed' from keras
def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.
    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = 'float32'
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                    tf.ones(shape, dtype=dtype),
                    tf.zeros(shape, dtype=dtype))


# ===========================
# ========= G A E ===========
# ===========================

class FactoredGatedAutoencoder:
    
    def __init__(self, numFactors, numHidden, learningRate=0.001, 
                 corrutionLevel=0.0,
                 weight_decay_vis=0.0,
                 weight_decay_map=0.0):
        """ Create a factored autoencoder
            tbd
        """
        self.F = int(numFactors)
        self.H = int(numHidden)
        self.learningRate = learningRate
        self.p = corrutionLevel
        self.eps_vis = weight_decay_vis
        self.eps_map = weight_decay_map
        
    def train(self, X, Y,
              epochs=150,
              batch_size=100,
              normalize_data=True):
        """ train the factored autoencoder
        X: x-input
        Y: y-input
        """
        F = self.F
        H = self.H
        p = self.p
        lr = self.learningRate
        eps_vis = self.eps_vis
        eps_map = self.eps_map
        
        n, dim = X.shape
        assert(dim == Y.shape[1])
        
        numpy_rng = np.random.RandomState(1)
        
        if normalize_data:
            X -= X.mean(0)[None, :]
            Y -= Y.mean(0)[None, :]
            X /= X.std(0)[None, :] + X.std() * 0.1
            Y /= Y.std(0)[None, :] + Y.std() * 0.1
        
        x = tf.placeholder(tf.float32, [None, dim])
        y = tf.placeholder(tf.float32, [None, dim])
        
        Wxf = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
        Wyf = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
        
        Whf = tf.Variable(np.exp(numpy_rng.uniform(
            low=-3.0, high=-2.0, size=(H, F)),dtype='float32'))
        Whf_in = tf.Variable(
            numpy_rng.uniform(
                low=-0.01, high=+0.01, size=(H, F)).astype('float32'))

        bmap = tf.Variable(np.zeros(H, dtype='float32'), name='bmap')
        bx = tf.Variable(np.zeros(dim, dtype='float32'), name='bx')
        by = tf.Variable(np.zeros(dim, dtype='float32'), name='by')


        x_corrupted = tf.multiply(random_binomial(tf.shape(x),p=p),x)
        y_corrupted = tf.multiply(random_binomial(tf.shape(y),p=p),y)
        
        fx = tf.matmul(x_corrupted , Wxf)
        fy = tf.matmul(y_corrupted , Wyf)
        
        mappings = tf.sigmoid(tf.matmul(tf.multiply(fx , fy), 
                                        tf.transpose(Whf_in)) + bmap)
        fH = tf.matmul(mappings, Whf)
        
        ox = tf.matmul(tf.multiply(fy , fH),tf.transpose(Wxf)) + bx
        oy = tf.matmul(tf.multiply(fx , fH),tf.transpose(Wyf)) + by
        
        cost = tf.nn.l2_loss(ox-x) + tf.nn.l2_loss(oy-y)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
        
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            for epoch in range(epochs):
                total_runs = int(n / epochs)
                print("total:", total_runs)
                for i in range(total_runs):
                    randidx = np.random.randint(
                        n, size=batch_size).astype('int32')
                    batch_xs = X[randidx]
                    batch_ys = Y[randidx]  
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                cost_ = sess.run(cost, feed_dict={x: X, y: Y}) / n
                print ("Epoch: %03d/%03d cost: %.9f" %\
                       (epoch,epochs ,cost_) ) 
            
        