import pickle
import os
import sys

size = int(sys.argv[1])
dir = '/mnt/disk1/tan_hm/corpus/train'
files = [os.path.join(dir, i) for i in os.listdir(dir)]

counter = 0
seg = 2**30
res = []

while size > 0:
    with open(files[counter]) as f:
        res.append(f.read(seg))
    counter += 1
    size -= 1

with open('/mnt/disk1/tan_hm/test_pkl.pkl', 'wb') as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
