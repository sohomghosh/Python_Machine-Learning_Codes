x = [1,2,3,2,2,2,3,4]
list(filter(lambda a: a != 2, x))
#[1, 3, 3, 4]
list(filter(lambda a: a != 2 and a!=3, x))
#[1, 4]
