import multiprocessing as mp
import time
import numpy as np

def func(x, a, b):
    return x*x

if __name__ == "__main__":   
    x = np.arange(int(1e6))

    start = time.time()
    results = [func(i, 1, 2) for i in x]
    print(time.time() - start)

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    results = pool.starmap(func,[(i, 1, 'as') for i in x])
    print(time.time() - start)
    pool.close()