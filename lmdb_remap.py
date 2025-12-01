import lmdb
import pickle
import zlib
import numpy as np
import os
import jax

def reorganize_lmdb(db_path, map_size = 1099511627776):
    env = lmdb.open(db_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        keys_to_remap = []
        for key, value in txn.cursor():
            if 700 <= pickle.loads(key)[0] <900:
                txn.delete(key)
            elif pickle.loads(key)[0] >=900:
                keys_to_remap.append((key, value))
       
        for key, value in keys_to_remap:
            ep, frame = pickle.loads(key)

            new_ep = ep - 200
            new_key = pickle.dumps((new_ep, frame))
            txn.delete(key)
            txn.put(new_key, value)
    
    env.close()
    print("原地重组完成！")
    

def verify_reorganization(db_path):
    env = lmdb.open(db_path, readonly=True, lock=False)
    
    episodes = set()
    total = set()
    
    with env.begin() as txn:
        frame_list = []
        for key, value in txn.cursor():
            ep, frame = pickle.loads(key)
            episodes.add(ep)
            total.add(pickle.loads(key))
        print(len(frame_list))
        print(len(total))
    
    env.close()
    

if __name__ == "__main__":
    map_size = 300*1024*1024