__all__=['a','b','data_reader']
#Submodules of empty_package will have a, b available
__version__ = '0.1'
__author__ = 'Sohom Ghosh'

from empty_package import a
from empty_package import b
from empty_package import data_reader

#If you remove the __init__.py file, Python will no longer look for submodules inside that directory, so attempts to import the module will fail.