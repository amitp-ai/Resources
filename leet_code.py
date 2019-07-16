#test
https://docs.python.org/3.6/distutils/setupscript.html
https://docs.python.org/3/tutorial/modules.html#packages
https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html

################
zza = 1

##################
def fib1(n):
    a,b = 0,1
    for i in range(1,n):
        temp = a+b
        a = b
        b = temp
    return b

#print(fib1(7))

##################
with open('data\data1.dat', 'r') as f:
    print(list(f.read()))

##################    
from setuptools import setup #superset of distutils

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="del_test_pkg",
    version="1.0",
    author="Example Author",
    long_description=long_description,
    packages=["del_test_pkg"],
    package_data = {'del_test_pkg': ['data/*.dat']},
    py_modules = ['fib', 'read_data'], #setuptools.find_packages()
)

##################
# Example Package

This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
