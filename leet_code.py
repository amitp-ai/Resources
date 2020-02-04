#test

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
