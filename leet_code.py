#test

"""
Syntax for decorators with parameters

def decorator(p):
    def inner_func():
        #do something
     return inner_func

@decorator(params)
def func_name():
    ''' Function implementation'''
The above code is equivalent to

def func_name():
    ''' Function implementation'''

func_name = (decorator(params))(func_name) #same as decorator(params)(func_name)

As the execution starts from left to right decorator(params) is called which returns a function object fun_obj. 
Using the fun_obj the call fun_obj(fun_name) is made. Inside the inner function, required operations are performed and 
the actual function reference is returned which will be assigned to func_name. Now, func_name() can be used to call the function 
with decorator applied on it.
"""

# Decorators with parameters in Python
def big_func(dumy):
    print(dumy)
    return func1

def func1(x):
    print('xxxxx')
    print(x)

big_func('zzzz')(2)

@big_func('zzz')
def func1(x):
    print(x)

print('1'*50)

def decorator(*args, **kwargs): 
    print("Inside decorator") 
    def inner(func): 
        print("Inside inner function") 
        print("I like", kwargs['like'])  
        return func 
    return inner 
@decorator(like = "geeksforgeeks") 
def func(): 
    print("Inside actual function") 
func() 

print('2'*50)

decorator(like = "geeksforgeeks")(func)()
