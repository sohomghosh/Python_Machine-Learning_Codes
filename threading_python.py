import threading

threads = 3   # Number of threads to create

def rev_str(a):
	print(a[::-1])


str_li=['hi','how','are','you']
jobs = []
j=0

for i in range(0, threads):
	out_list = list()
	thread = threading.Thread(target=rev_str(str_li[j]))
	j=j+1
	jobs.append(thread)


# Start the threads
for j in jobs:
	j.start()

# Check all of the threads have finished
for j in jobs:
	j.join()


#References
#https://www.quantstart.com/articles/parallelising-python-with-threading-and-multiprocessing
#https://chitcode.wordpress.com/2014/06/20/feature-extraction-using-python-multi-threading-and-multi-processing/
#https://docs.python.org/3/library/multiprocessing.html
#https://www.python-course.eu/threads.php
