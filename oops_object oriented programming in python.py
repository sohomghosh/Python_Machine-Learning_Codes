#Without using class, python code saved as "program_name.py"
import <program_name>
#Access function by program_name.function_name()


#With class, python code saved as "program_name.py" having class named "class_name"
from <program_name.py> import <class_name>




##Source: https://stackoverflow.com/questions/625083/python-init-and-self-what-do-they-do
###In this code:


class A(object):
	def __init__(self):
		self.x = 'Hello'

	def method_a(self, foo):
		print self.x + ' ' + foo


#the self variable represents the instance of the object itself. Most object-oriented languages pass this as a hidden parameter to the methods defined on an object; Python does not. You have to declare it explicitly. When you create an instance of the A class and call its methods, it will be passed automatically, as in

a = A()               # We do not pass any argument to the __init__ method
a.method_a('Sailor!') # We only pass a single argument


#The __init__ method is roughly what represents a constructor in Python. When you call A() Python creates an object for you, and passes it as the first parameter to the __init__ method. Any additional parameters (e.g., A(24, 'Hello')) will also get passed as arguments--in this case causing an exception to be raised, since the constructor isn't expecting them.



####################### REAL LIFE EXAMPLE ###############################
#Reference-1: https://github.com/HackerEarth-Challenges/machine-learning-challenge-2/blob/master/Rank_1_Roman/mean_evaluation.py
#Reference-2: https://github.com/HackerEarth-Challenges/machine-learning-challenge-2/blob/master/Rank_1_Roman/first%20model.ipynb

####Reference-2 calling Reference-1

##Reference-1 saved as "mean_evaluation.py" is as follows:
import os
class roman_mean:
    def __init__(self, directory, data, target, n_folds_gen, n_folds_sub, seed, sub_seed, ltr,
                 extra_train = None, extra_target = None):
        self.directory = directory
        self.n_folds_gen = n_folds_gen
        self.n_folds_sub = n_folds_sub
        self.seed = seed
        self.sub_seed = sub_seed
        self.ltr = ltr
        self.data = data
        self.target = target
        self.extra_train = extra_train
        self.extra_target = extra_target
    
    def save_in_file(self, data):
        for x in data.columns.values:
            directory = self.directory + '\\features\\' + x
            print(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                print(x + ' already save.')
                continue
            data.loc[:, x].to_csv(directory + '\\' + x + '.csv', index = None, header = True)


##Calling Reference-1 from Reference-2; Reference-2 is as follows
from mean_evaluation import roman_mean
import pandas as pd

roman_model = roman_mean(directory = '/home/data/', n_folds_gen = 10, n_folds_sub = 5, seed = 322, sub_seed = 228, ltr = ltr, data = data, target = final_status)

##Do operations on "roman_model" i.e. do training by passing the dataset taken as input from local system in Reference-2  by calling "roman_model"

#Saving roman_model
roman_model.save_in_file(data)

#Also check this
#https://www.hackerearth.com/practice/python/object-oriented-programming/classes-and-objects-i/tutorial/
#https://www.hackerearth.com/practice/python/object-oriented-programming/classes-and-objects-ii-inheritance-and-composition/tutorial/

#Inheritance
class DerivedClassName(BaseClassName):
    pass


#Composition
class GenericClass:
    define some attributes and methods

class ASpecificClass:
    Instance_variable_of_generic_class = GenericClass

# use this instance somewhere in the class
    some_method(Instance_variable_of_generic_class)


	
#Inheritance: create a child cass of a base class
#Compostion: create a object instance of a class within another class

