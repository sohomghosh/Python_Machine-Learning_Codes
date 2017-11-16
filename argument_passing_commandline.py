import sys
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))
print("The 3rd argument is: " , sys.argv[3])

#execute
#python3 argument_passing_commandline.py 3 'abc' 356 'ffd'

#OUTPUT
#This is the name of the script:  argument_passing_commandline.py
#Number of arguments:  5
#The arguments are:  ['argument_passing_commandline.py', '3', 'abc', '356', 'ffd']
#The 3rd argument is:  356
