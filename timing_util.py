from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        elapsed_time = end - start
        print('Elapsed time: {}'.format(elapsed_time) + ' for ' + str(f.__name__))
        return result
    return wrapper