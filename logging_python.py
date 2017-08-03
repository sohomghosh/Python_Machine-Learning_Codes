import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

#open file example.log and the following lines would be found
'''
DEBUG:root:This message should go to the log file
INFO:root:So should this
WARNING:root:And this, too
'''
