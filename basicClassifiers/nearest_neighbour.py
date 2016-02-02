import tensorflow as tf
import numpy as np

import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot="True")

# Limiting the mnist data
Xtr,Ytr=mnist.train.next_batch(5000)
Xte,Yte=mnist.test.next_batch(200)

# Reshanping Images to 1D

Xtr=np.reshape(Xtr,newshape=(-1,28*28))
Xte=np.reshape(Xte,newshape=(-1,28*28))


# Graph input

xtr=tf.placeholder("float",[None,784])
xte=tf.placeholder("float",[784])


# nearest neighbour calculation using L1 Distance
# Calculate L1 distance

distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.neg(xte))),reduction_indices=1)

pred=tf.arg_min(distance,0)

accuracy =0.

# initializing hte variables
init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    
    
    # loop over test data
    for i in range(len(Xte)):
        nn_index=sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        # get nearest neighbout calss label as compared to its true class label
        print "Test", i , " Prediction: ",np.argmax(Ytr[nn_index]),"True Class :",np.argmax(Yte[i])
        
        # calculate Accuracy
        if np.argmax(Ytr[nn_index])==np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
        
    
    print "Done!"
    print "Accuracy: ", accuracy
