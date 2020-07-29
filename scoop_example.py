import random
from scoop import futures, shared
import scoop
from logging_setup import logger

def abs_mult_x(a, x):
    return (a, abs(a) * x)

def abs_mult_x_plus_y(a, x, y):
    return (a, abs(a) * x + y)

def abs_mult_x_plus_const(a, x):
    logger.warning("Using shared seed")
    return (a, abs(a) * x + shared.getConst('seed'))


if __name__ == '__main__':
    shared.setConst(seed=100)
    data = [random.randint(0, 1000) for r in range(100)]
    #dp = list(futures.map(abs_mult_x, data, [0.5 for i in range(len(data))]))
    #dp = list(futures.map(abs_mult_x_plus_y, data, [0.5 for i in range(len(data))], [1.5 for i in range(len(data))]))
    dp = list(futures.map(abs_mult_x_plus_const, data, [0.5 for i in range(len(data))]))
    print(dp)
