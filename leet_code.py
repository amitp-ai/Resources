#test

"""
#TENSORFLOW BASICS
#Resource: https://www.tensorflow.org/guide/variables
#Resource: https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow


1. tf.Variable() vs tf.get_variable()
    Two main differences:
        1. tf.Variable will always create a new variable (even if you give it a name, see below), 
        whereas tf.get_variable gets from the graph an existing variable 
        with those parameters, and if it does not exists, it creates a new one.
        2. tf.Variable requires that an initial value be specified.

    with tf.variable_scope("one"):
        a = tf.get_variable("v", [1]) #a.name == "one/v:0"
    with tf.variable_scope("one"):
        b = tf.get_variable("v", [1]) #ValueError: Variable one/v already exists
    with tf.variable_scope("one", reuse = True):
        c = tf.get_variable("v", [1]) #c.name == "one/v:0"

    with tf.variable_scope("two"):
        d = tf.get_variable("v", [1]) #d.name == "two/v:0"
        e = tf.Variable(1, name = "v", expected_shape = [1]) #e.name == "two/v_1:0"
        #If the default graph already contains an operation/variable named "v", 
        #then TensorFlow would append "_1", "_2", and so on to the name, in order to make it unique.
        #see here: https://www.tensorflow.org/guide/graphs#naming_operations
        
    assert(a is c)  #Assertion is true, they refer to the same object.
    assert(a is d)  #AssertionError: they are different objects
    assert(d is e)  #AssertionError: they are different objects


2. To get a list of all tensorflow variables
    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

3. Why create variable names in tensorflow:
    1. when saving/restoring variables, it saves variables to the tensorflow namespaces and not the python namespace.
    2. for tensorboard, the variable names shown are from the tensorboard namespace
    3. python vs tensorflow namespace. For example:
    v = tf.get_variable(name="v1", shape=[2], initializer=tf.constant([1,1]))
    v = tf.get_variable(name="v2", shape=[2], initializer=tf.constant([2,2]))
    We now have two tensorflow variables v1 and v2, a single python variable v.
    The python variable v is just like a pointer, as is the case with Python variables.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(v)) #[2,2]
        print(sess.run(fetches='v1:0')) #[1,1]
        print(sess.run(fetches='v2:0')) #[2,2]

4. tf.get_variable() initialization:
    The initializer argument has to be either a TensorFlow Tensor object (which can be constructed by calling tf.constant on a numpy value), or a 'callable' that takes two arguments, shape and dtype, i.e. the shape and data type of the value that it's supposed to return.
    Example:
    init = tf.constant(np.random.rand(1, 2))
    tf.get_variable('var_name', initializer=init)
    
    init = lambda shape, dtype: np.random.rand(*shape)
    tf.tf.get_variable('var_name', initializer=init, shape=[1, 2])

    value = [0, 1, 2, 3, 4, 5, 6, 7]
    # value = np.array(value)
    init = tf.constant_initializer(value)
    tf.tf.get_variable('var_name', initializer=init, shape=[1, 2])


5. TF variable sharing and name scoping
"""
The method tf.get_variable can be used with the name of the variable as the argument to either create a new variable with such name or retrieve the one that was created before. This is different from using the tf.Variable constructor which will create a new variable every time it is called (and potentially add a suffix to the variable name if a variable with such name already exists).
It is for the purpose of the variable sharing mechanism that a separate type of scope (variable scope) was introduced.
As a result, we end up having two different types of scopes:

    name scope, created using tf.name_scope
    variable scope, created using tf.variable_scope

Both scopes have the same effect on all operations as well as variables created using tf.Variable, i.e., the scope will be added as a prefix to the operation or variable name.
However, name scope is ignored by tf.get_variable. We can see that in the following example:

"""
tf.reset_default_graph()
with tf.name_scope('train'): #name_scope is ignored by tf.get_variable() but not tf.variable()
    a = tf.get_variable('tfa', [1]) #a.name is 'tfa:0'
    a2 = tf.Variable([1,2], name='tfa2') #a2.name is 'train/tfa2:0'
with tf.variable_scope('model'):
    b = tf.get_variable('tfb', [1]) #b.name is 'model/tfb:0'


#Sharing variables
tf.reset_default_graph()
with tf.variable_scope('model1', reuse=None):
    a = tf.get_variable('v', [1,2])
    
with tf.variable_scope('model1', reuse=True):
    b = tf.get_variable('v', [1,2])
    #c = tf.get_variable('w', [1,2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([a]))
    print(sess.run([b]))
    #print(sess.run([c]))


6. To find name of a tensorflow variable: do 'python_namespace_name.name' e.g. b.name in the example above.

7. # Graph definition
graph = tf.Graph()
with graph.as_default():
  # Operations created in this scope will be added to `graph`.
  c = tf.constant("Node in graph")

# Session
with tf.Session(graph=graph) as sess:
    print(sess.run(c).decode("utf8"))

a = np.array([1,2])
b = np.array([11,22])
c = np.array([111,222])
tf.reset_default_graph()
with tf.name_scope(None, 'train', [a,b]):
    aa = tf.convert_to_tensor(a, name="tfa", dtype=tf.int32)
    bb = tf.convert_to_tensor(b, name="tfb", dtype=tf.int32)
    cc = tf.convert_to_tensor(c, name="tfc", dtype=tf.int32)
print(aa.name, bb.name, cc.name)


8. Variable reusing

    with tf.variable_scope('train', reuse=None):
        a = tf.get_variable('tfa', [2])
        b = tf.get_variable('tfa', [2]) #this fails b'cse reuse is not True

    tf.reset_default_graph()
    b = []
    with tf.variable_scope('train', reuse=False):
        for i in range(3):
            if i > 0: tf.get_variable_scope().reuse_variables()
            a = tf.get_variable('tfa', [1,2])
            b.append(a)
        c = tf.concat(0, b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))

9. To get shape of a tensorflow tensor a, do 'a.get_shape'

10. Update a tf variable
    tf.reset_default_graph()
    with tf.variable_scope('model', reuse=False):
        w = tf.get_variable('w', [1,2])
        #w = tf.assign(w, tf.constant([[1.0,1]]))
        update = w.assign(tf.constant([[1.0,1]]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(w))
        sess.run(update)
        print(sess.run(w))

"""
