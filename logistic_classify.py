#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]


# In[3]:


X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# In[4]:


W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# In[5]:


hypo = tf.sigmoid(tf.matmul(X,W)+b)


# In[10]:


cost = -tf.reduce_mean(Y*tf.log(hypo)+(1-Y)* tf.log(1-hypo))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# In[11]:


pred = tf.cast(hypo > 0.5 , dtype=tf.float32)
accur = tf.reduce_mean(tf.cast(tf.equal(pred,Y),dtype=tf.float32))


# In[12]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _ = sess.run([cost,train],feed_dict={X:x_data, Y:y_data})
    if step %200 == 0:
        print(step, cost_val)


# In[13]:


h,c,a= sess.run([hypo, pred, accur], feed_dict={X:x_data, Y:y_data})
print("\nhypothesis : ",h,"\nCorrect (Y): ", c, "\Accuracy: ",a)


# In[15]:


print(sess.run(hypo, feed_dict={X:[[0,0], [1,9], [3,5]]}))


# In[ ]:




