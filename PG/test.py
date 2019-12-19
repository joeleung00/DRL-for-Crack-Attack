import multiprocessing
from os import getpid

def worker(tmp):
    print('I am number %d in process %d' % (1, getpid()))
    return [getpid()] * 2

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 2)
    print(pool.map(worker, range(5)))
