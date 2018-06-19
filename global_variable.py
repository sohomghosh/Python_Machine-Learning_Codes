#global variable pyhton refer
global flag
flag = 0

#global variable refer within a function so that its value remians same during multiple function call
def func():
  global flag
  flag = 0
  #write codes here
