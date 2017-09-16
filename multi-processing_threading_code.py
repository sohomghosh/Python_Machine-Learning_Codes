from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(15) #replace 15 by number of threads you want to use

 def fnc(a):
     return (a*10)

results = pool.map(fnc, [10,20])
