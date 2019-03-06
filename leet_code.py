# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:17:12 2016

@author: amit_p
"""

#y_train_num_each_classes = num_of_each_classes(y_train) #array of the number of each class
#
#final_num_data_per_class = 5000
#num_fake_data = final_num_data_per_class - y_train_num_each_classes
#
#


'''
# Distribution transformation using non-linear functions
import numpy as np
import matplotlib.pyplot as plt

x_dist = []
y_dist = []

for i in range(1000):
    x = np.random.normal(0,10)
    x_dist.append(x)
    y_dist.append(100*x**2)

_, axs = plt.subplots(2, 2)
#print(axs.shape)
axs[0,0].plot(x_dist)
axs[0,1].plot(y_dist)
axs[1,0].hist(x_dist, bins=100)
axs[1,1].hist(y_dist, bins=100)
# /End distribution transformation using non-linear functions



import numpy as np
import tensorflow as tf

X = np.random.random((2,3))
X_tf = tf.placeholder(tf.float64, shape=[None, 3])

X_dmy = X_tf[0:-1,:] #tf.slice(X_tf, [1,0], [-1,-1])
X_dmy = tf.pad(X_dmy, [[1,0],[0,0]])

wt = np.random.random((2,3))
wt_tf = tf.Variable(wt)
wt_tf = tf.transpose(wt_tf)

bs = np.random.random((2)) #(5,1) doesn't broadcast. needs to be (1,5) for relu_layer
#bs = np.random.random((1,5)) #(5,1) doesn't broadcast. needs to be (1,5) for relu
bs_tf = tf.Variable(bs)

y_pred = tf.add(tf.matmul(X_tf, wt_tf), bs_tf) #due to broadcasting it works

grads = tf.gradients(y_pred, X_tf)

y_pred = tf.nn.relu(y_pred)
y_pred_relu = tf.nn.relu_layer(X_tf, wt_tf, bs_tf)

y_pred_norm = tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, y_pred)))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    y_prediction = session.run(y_pred, feed_dict = {X_tf:X})
    y_prediction_relu = session.run(y_pred_relu, feed_dict = {X_tf:X})
    #gradients = session.run(grads, feed_dict = {X_tf:X})
    yp_nm = session.run(X_dmy, feed_dict = {X_tf:X})


print(y_prediction)
print()
print(y_prediction_relu)
print()
#print(gradients)
print(yp_nm)
print()
print(X)




#*******#2 tensorflow***********************

import numpy as np
import tensorflow as tf

n1 = np.array(
[[[1, 2, 3],
[4,5,6]],
[[7,8,9],
[10,11,12]]])

t1 = tf.Variable(n1)

print(t1.get_shape().ndims)
print(t1.get_shape().as_list())
print(t1.get_shape())

print(n1.shape)

t2 = tf.transpose(t1, perm=[0,2,1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ntf2 = sess.run(t2)

#Tensorflow Namespaces
#Namespaces is a way to organize names for variables and operators in hierarchical manner (e.g. "scopeA/scopeB/scopeC/op1")
#
#tf.name_scope creates namespace for operators in the default graph.
#tf.variable_scope creates namespace for both variables and operators in the default graph.
#tf.op_scope same as tf.name_scope, but for the graph in which specified variables were created.
#tf.variable_op_scope same as tf.variable_scope, but for the graph in which specified variables were created.
#################################
    
#######3. Decorators############
import time


def timing_function(some_function):

    """
    Outputs the time a function takes
    to execute.
    """

    def wrapper(wrapper_var2):
        wrapper_var2 *= 1
        t1 = time.time()
        wrapper_var1 = 10
        some_function(wrapper_var1) #this is passed to var1 in my_function() function
        t2 = time.time()
        return "Time it took to run the function: " + str((t2 - t1)) + "\n"
    return wrapper #return the wrapper function


@timing_function
def my_function(var1):
    var1 = var1+0
    num_list = []
    for num in (range(0, 10000)):
        num_list.append(num)
    print("\nSum of all the numbers: " + str((sum(num_list))))

print(my_function(11)) #11 is passed to wrapper_var2 in wrapper() function
#The above @syntax is same as running the following: 
    #my_function = timing_function(my_function)
    #print(my_function(11))

#################################

import tensorflow as tf
import numpy as np

in1_tf = tf.placeholder(dtype=tf.float32, shape=[None, 38,38,6])
in2_tf = tf.placeholder(dtype=tf.float32, shape=[None, 38,38,6])
in_tf = (in1_tf, in2_tf)
out_tf = in_tf[0] + in_tf[1]

in1_np = np.zeros(shape=[2,38,38,6])
in2_np = np.ones(shape=[2,38,38,6])
in_np = (in1_np, in2_np)


with tf.Session() as sess:
    out_np = sess.run(out_tf, feed_dict={in_tf:in_np})



#################################
#USING FOR LOOP
import random
tstrt = time.time()
N=int(1e6)
mysum=0
for ii in range(N):
    mysum += 1/(random.uniform(0.0001,1))
    #print(random.uniform(0,1))

sample_mean = mysum/N
print(time.time()-tstrt)
print(sample_mean)

#######################
#WITH TENSORFLOW (need to fix this it doesn't work)
import tensorflow as tf
import numpy as np
import random
N=1.0e4
tN = tf.Variable(N)
mysum=0.0
tsum = tf.Variable(mysum)
for ii in range(int(N)):
    tsum = tsum + 1/(random.uniform(0,1))

tsample_mean = tsum/tN

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sample_mean = sess.run(tsample_mean)
    
print(sample_mean)
#######################

#Using Numpy
import time
tstrt = time.time()
N = int(1e6)
running_sample = 0
num_iter = 1
for _ in range(num_iter):
    random_mat = np.random.uniform(0.0001,1,size=[N,1])
    random_mat = 1 / random_mat
    sample_mean = np.sum(random_mat)/(N*1)
    running_sample += sample_mean
    
print(time.time()-tstrt)
print(running_sample/num_iter)


from sys import getsizeof
print(getsizeof(random_mat)/1e9, "GB")


##########MARKOV CHAIN##########################
import numpy as np
p0 = np.array([[0.15],[0.85]])
T = np.array([[0.2,0.4],[0.8,0.4]])

for i in range(20):
    p0 = np.matmul(T,p0)
    p0 = p0/np.sum(p0) #normalize into a probability
    print(p0)

####################################################


#####REALTIME NOISE ANALYSIS#############
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack

pi = np.pi
t = np.arange(0,10e-3,10e-6)
noise_sum = np.zeros_like(t)
num_samples = 100
start_freq = 1e3
stop_freq = 10e3
for _ in range(num_samples):
    amp = np.random.random_sample()*4e-9*np.sqrt(1e3)
    freq = 3e3 #np.random.random_sample()*(stop_freq-start_freq) + start_freq
    noise_sum += 0 + amp*np.cos(2*pi*freq*t + 0)

plt.plot(t,noise_sum)
#plt.show()
#plt.hist(noise_sum)

############################

#import numpy as np
#import matplotlib.pyplot as plt
#import scipy.fftpack
#
## Number of samplepoints
#N = 600
## sample spacing
#T = 1.0 / 800.0
#x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#yf = scipy.fftpack.fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#fig, ax = plt.subplots()
#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
#plt.show()
#
#plt.plot(x,y)


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 1000 #N = tstop/rtnstep
# sample spacing
T = 10e-6 #T=rtn_step
x = np.linspace(0.0, N*T, num=N)
y = np.zeros_like(x)
for i in range(np.shape(y)[0]):
    y[i] = 4e-9*np.sqrt(1/T) * np.random.standard_normal() #bw = 1/T

#y = 10*np.sin(1/(10*T) * 2.0*np.pi*x)  
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

plt.plot(x,y)


#PLOT A SINGLE FREQ. SINE WAVE WITH AMPLITUDE NOISE
N = 1000
T = 10
freq = 1/(100*T)
x = np.linspace(0.0, N*T, num=N)
y = np.zeros_like(x)

num_samples=10
sub_sig_len = int(N/num_samples)
for run in range(num_samples):
    strt = run*sub_sig_len
    stp = strt+sub_sig_len
    amp = 4 * np.mod(run,2) #* np.random.random_sample() #np.random.standard_normal()
    if(run > 0):
        ph = 0 #np.arcsin(y[strt-1]/amp) - 2*np.pi*freq*x[strt]
    else:
        ph = 0
    y[strt:stp] = amp*np.sin(2*np.pi*freq*x[strt:stp] + ph)

#y = amp*np.sin(2*np.pi*freq*x)    
plt.plot(x,y)

#######################################################

list_steps = []
def factorial(n):
    if n<=1:
        factrl=1
        list_steps.append(1)
    else:
        factrl = n*factorial(n-1)
        list_steps.append(1)

    return factrl

#######################################################

########## MCTS #################
#pseudo code

Node_List = [] #list of nodes for back propagation (initialize at the beginning of each turn)
def MCTS(N):
    if N is fully expanded: #i.e. all of N's children are visited
        node_to_try = UCT(childern of N)
        MCTS(node_to_try) #recurse
        Node_List.append(node_to_try)
    else:
        next_rode = randomly pick one of N's unvisited children
        reward = monte_carlo_sim(next_node) #using random policy as the default policy
        Node_List.append(next_node)
    
    return None
backpropagate(reward,node_list) #backpropagate the rewards to all the relevant nodes and updated their statistics

        
#################################

a = 0
def test_func():
    global a #to tell it to use global a (so it can both access and modify a)
    b = a
    b += 3
    a = b
    print(b)

test_func()
test_func()
test_func()
print(a)

####################
#Tensorflow
import numpy as np
import tensorflow as tf

np.random.seed(10)
X = np.random.random((2,3))
X_tf = tf.placeholder(tf.float64, shape=[None, 3])

wt = np.random.random((2,3))
wt_tf = tf.Variable(wt)
wt_tf = tf.transpose(wt_tf)

bs = np.random.random((2)) #(5,1) doesn't broadcast. needs to be (1,5) for relu_layer
#bs = np.random.random((1,5)) #(5,1) doesn't broadcast. needs to be (1,5) for relu
bs_tf = tf.Variable(bs)

y_pred = tf.add(tf.matmul(X_tf, wt_tf), bs_tf) #due to broadcasting it works

y_pred = tf.nn.relu(y_pred)

y_pred_norm = tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, y_pred)))


yGT = np.array([[0,1],[1,0]])
yGT_tf = tf.placeholder(tf.float64, shape=[None,2])

sum_ypred_tf = tf.reduce_sum(y_pred, axis=1)
print(sum_ypred_tf.get_shape())

loss_tf = tf.nn.softmax_cross_entropy_with_logits(y_pred, yGT_tf)
print(loss_tf.get_shape())

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    y_prediction,lbl,loss,sum_ypred = session.run([y_pred,yGT_tf,loss_tf,sum_ypred_tf], feed_dict = {X_tf:X, yGT_tf:yGT})


print(y_prediction)
print(lbl)
print(loss)
print('sum ypred', sum_ypred)


loss = -np.log(np.exp(y_prediction[1,0])/(np.exp(y_prediction[1,0]) + np.exp(y_prediction[1,1])))
print(loss)

a = tf.Variable([[1,2,3],[4,5,6]])
print(a.get_shape())
b = tf.Variable([[10,20,30]])
a = tf.multiply(b,a)
print(a.get_shape())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_np = sess.run(a)
    
print(a_np)


import tensorflow as tf
i = tf.constant(0)
c = lambda i: tf.less(i, [10,5])[0]
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
f = tf.less(i, 10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r_np, f_np = sess.run([r, f])
    
print(r_np, f_np)
    
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

#x_gt = np.reshape(np.arange(-5,5,0.1), (-1,1))
#y_gt = np.sin(2*x_gt)
x_gt = [[-5],[-4],[-3],[-2],[-1.9],[-1],[0],[1],[1.9],[2],[3],[4],[5]]
y_gt = [[5],[5],[5],[5],[0],[0],[0],[0],[0],[5],[5],[5],[5]]
#x_gt = [[-8],[-7],[-6],[-5.1],[-5],[-4],[-3],[-2],[-1.9],[-1],[0],[1],[1.9],[2],[3],[4],[5],[5.1],[6],[7],[8]]
#y_gt = [[0],[0],[0],[0],[5],[5],[5],[5],[0],[0],[0],[0],[0],[5],[5],[5],[5], [0], [0], [0], [0]]
#x_gt = [[-5],[0],[5]]
#y_gt = [[-5],[0],[5]]
#x_gt = np.arange(-5,5,0.1).reshape(-1,1)
#y_gt = x_gt**3 + x_gt**2 + x_gt + 1
x_test = np.arange(-8,8,0.1).reshape(-1,1)

x_tf = tf.placeholder(tf.float64, shape=[None,1])
y_tf = tf.placeholder(tf.float64, shape=[None,1])
keep_prob = tf.placeholder(tf.float64)
bn_phase = tf.placeholder(tf.bool, name='bn_phase')

reg_factor = 1e-6 #-2

num_h1 = 400
np.random.seed(0)
w1 = np.random.random((1,num_h1))*1/num_h1
w1 = tf.Variable(w1, name="w1")
b1 = np.zeros((num_h1))
b1 = tf.Variable(b1, name="b1")

num_h2 = 200
w2 = np.random.random((num_h1,num_h2))*1/num_h2
w2 = tf.Variable(w2, name="w2")
b2 = np.zeros((num_h2))
b2 = tf.Variable(b2, name="b2")


w3 = np.random.random((num_h2,1))*1/num_h2
w3 = tf.Variable(w3, name="w3")
b3 = np.zeros((1))
b3 = tf.Variable(b3, name="b3")

h1 = tf.add(tf.matmul(x_tf, w1, name="matmul1"), b1) #Nxh1
h1 = tf.nn.elu(h1)
#h1 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=bn_phase, scope='bn')
h1 = tf.nn.dropout(h1, keep_prob)

h2 = tf.add(tf.matmul(h1, w2, name="matmul2"), b2) #Nxh2
h2 = tf.nn.elu(h2)
#h2 = tf.contrib.layers.batch_norm(h2, center=True, scale=True, is_training=bn_phase, scope='bn')
h2 = tf.nn.dropout(h2, keep_prob)
y_pred = tf.add(tf.matmul(h2, w3, name="matmul3"), b3) #Nx1

loss_v_tf = (y_pred-y_tf)**2
loss_s_tf = tf.reduce_mean(loss_v_tf)
reg_loss = tf.reduce_sum(w1**2) + tf.reduce_sum(b1**2) + \
            tf.reduce_sum(w2**2) + tf.reduce_sum(b2**2) + \
            tf.reduce_sum(w3**2) + tf.reduce_sum(b3**2)

loss_s_tf += reg_factor*reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
train_optimizer = optimizer.minimize(loss_s_tf)

'''
#find gradients
a = tf.constant(10)
b = tf.Variable(2)
c = 2*a+b
#grads_tf = tf.gradients(c, [a,b])
grads_tf = tf.gradients(c, a)

grads_pred_y = tf.gradients(y_pred, x_tf)

a = tf.Variable([1,2])
c = tf.reduce_sum(5*a)
#hess_tf = tf.hessians(c, a)

eps = 0.001
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    '''
    grads_np = sess.run(grads_tf)
    print(grads_np)
    '''
    
    for epoch in range(30000):
        loss_s_np, _ = sess.run([loss_s_tf,train_optimizer], feed_dict={x_tf: x_gt, y_tf: y_gt, keep_prob: 1.0, bn_phase: True})
        print(epoch, ': ', loss_s_np)
    
    y_pred_np = sess.run(y_pred, feed_dict={x_tf: x_gt, keep_prob: 1.0, bn_phase: False})
    y_test_np = sess.run(y_pred, feed_dict={x_tf: x_test, keep_prob: 1.0, bn_phase: False})
    
    '''
    #find the argmin of y_pred
    x_min = -1
    for _ in range(200):
        grads_np, y_pred_np_0 = sess.run([grads_pred_y, y_pred], feed_dict={x_tf: np.array([[x_min]]), keep_prob: 1.0, bn_phase: False})
        _, y_pred_np_1 = sess.run([grads_pred_y, y_pred], feed_dict={x_tf: np.array([[x_min+eps]]), keep_prob: 1.0, bn_phase: False})
        num_grad = (y_pred_np_1-y_pred_np_0)/eps
        print(x_min, grads_np, num_grad)
        x_min = x_min - 0.1*grads_np[0][0,0]
    '''

plt.plot(x_gt, y_gt)
plt.plot(x_gt, y_pred_np)
#plt.plot(x_test, y_test_np)

#elu seems to work better then relu in terms of training time


