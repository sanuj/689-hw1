import random
import sys

data_file = open(sys.argv[1], 'w')

size = int(sys.argv[2])
ranges = [(1, 10), (10, 100), (100, 1000)]

for r in ranges:
    for i in range(size):
        data = random.randint(*r)
        data_file.write(','.join((str(data), str(data+1), str(data+2), str(data+3))))
        data_file.write('\n')

data_file.close()
