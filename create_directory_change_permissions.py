import os
directory = 'abracadabra'
if not os.path.exists(directory):
    os.makedirs(directory)
    os.chmod(directory,  0o666) #os.chmod(directory,  0o644)


