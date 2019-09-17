#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x_data = [[73.,80.,75.], [93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
# y_data = [[152.],[185.],[180.],[196],[142]]
y_data = [[228.],[274.],[270.],[294.],[209]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = tf.matmul(X,W)+b
cost = tf.reduce_mean(tf.square(hypo - Y))


# In[3]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# In[4]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[5]:


for step in range(20001):
    cost_val, hy_val, _ = sess.run([cost, hypo, train], feed_dict={X:x_data, Y:y_data})
    if step % 100 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


# In[6]:


print("X: [100, 50, 90] , Y: ",sess.run(hypo, feed_dict={X:[[100.,50.,90.]]}))

