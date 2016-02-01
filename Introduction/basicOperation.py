import tensorflow as tf

a=tf.constant(2)
b=tf.constant(3)

with tf.Session() as sess:
    print "a=2, b=3"
    print "addition of constants: %i " % sess.run(a+b)
    print "Multiplication of constants: %i " % sess.run(a*b)
    
a=tf.placeholder(tf.types.int16)
b=tf.placeholder(tf.types.int16)


add=tf.add(a,b)
mul=tf.mul(a,b)

with tf.Session() as sess:
    print "Additon is %i" % sess.run(add,feed_dict={a:2,b:3})
    print "Multiplication is %i" % sess.run(mul,feed_dict={a:2,b:3})
    
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
product =tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    print sess.run(matrix1)
    print sess.run(matrix2)
    print sess.run(product)