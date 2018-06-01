import re
my_string = 'hi how are you doing'
re.sub(r'([^\s\w]|_)+', '', my_string)
