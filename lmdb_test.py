import lmdb
import pickle
import zlib
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import time
# import jax
# import jax.numpy as jnp

class LMDBTensorDataset: #
    def __init__(self, db_path: str, keys_list = None):
        self.db_path = db_path
        self.env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, meminit=False)
    
    def __getitem__(self, key):
        # env = lmdb.open(str(self.db_path), readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            mask = 0
            feature = None
            key_bytes = pickle.dumps(key)
            tensor_bytes = txn.get(key_bytes)

            if tensor_bytes is not None:
                tensor_bytes = zlib.decompress(tensor_bytes)
                # 将 PyTorch Tensor 转换为 JAX Array
                torch_tensor = pickle.loads(tensor_bytes)
                
                if hasattr(torch_tensor, 'numpy'):  # 如果是 PyTorch Tensor
                    numpy_array = torch_tensor.detach().cpu().to(torch.float16).numpy()
                    feature = jnp.array(numpy_array, dtype=jnp.bfloat16)
                elif hasattr(torch_tensor, '__jax_array__') or isinstance(torch_tensor, jax.Array):  # 如果是 JAX Array
                    feature = torch_tensor  # 直接赋值
                else:  # 如果是 numpy array 或其他
                    numpy_array = torch_tensor
                    feature = jnp.array(numpy_array, dtype=jnp.bfloat16)
                mask = 1
        # env.close()
        return feature, mask
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def __del__(self):
        self.close()

class BatchedLMDBDataset(LMDBTensorDataset):
    def __init__(self, db_path: str, keys_list = None): 
        super().__init__(db_path, keys_list)
    
    def __getitem__(self, keys):
        # 批量读取
        batch_features = []
        batch_masks = []
        
        for key in keys:
            feature, mask = super().__getitem__(key)
            
            if feature is not None:
                batch_features.append(feature)
            else:
                batch_features.append(jax.numpy.zeros((2*37*37,2048)))
            
            batch_masks.append(mask)
        
        if batch_features:
            batch_tensors = jnp.stack(batch_features, axis=0)
        else:
            # batch_tensors = jnp.array([])
            batch_tensors = None
        
        batch_mask = jnp.array(batch_masks, dtype=jnp.int32)
        
        return batch_tensors, batch_mask
    

def create_lmdb_database(token_path, db_path:str, map_size=70*1024*1024*1024):  
    """创建LMDB数据库"""    
    # remove existing database if exists
    # if os.path.exists(db_path):
    #     print(f"Removing existing database at {db_path}")
    #     for file in os.listdir(db_path):
    #         os.remove(os.path.join(db_path, file))

    env = lmdb.open(db_path, map_size=map_size)
    
    for episode_idx in range(len(os.listdir(token_path))):
        sub_dir=os.path.join(token_path,"ep"+str(episode_idx+1).zfill(6))
        fp=os.path.join(sub_dir,"vggt.npz")
        if os.path.isdir(sub_dir):
            data=np.load(fp) 
            tokens=torch.from_numpy(data["feature"]).to(dtype=torch.bfloat16) #shape=(2, 1, 16, 1374, 2048)
            tokens=torch.squeeze(tokens[...,5:,:],1) #shape=(2, 16, 1374, 2048)
            tokens=tokens.transpose(1, 0) #shape=(16, 2, 1374, 2048)
            
            fns=data["sampled_file_list"]
            with env.begin(write=True) as txn:
                for i, im_name in enumerate(fns[0]):
                    strings=im_name.split('/')
                    frame_index=int(strings[-1][4:10])
                    key_bytes = pickle.dumps((episode_idx,frame_index-1))
                    tensor_bytes = pickle.dumps(tokens[i].reshape(-1,2048))
                    compressed = zlib.compress(tensor_bytes, level=1)
                    txn.put(key_bytes, compressed)
            # if episode_idx%100==0:
                print(f"Loaded vggt tokens for episode {episode_idx} from {fp}")
    env.close()
    
 
    print(f"数据库创建完成: {db_path}")
    
    
def create_lmdb_database_visual(db_path:str, map_size=70*1024*1024*1024):  
    """创建LMDB数据库"""    
    # remove existing database if exists
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        for file in os.listdir(db_path):
            os.remove(os.path.join(db_path, file))

    env = lmdb.open(db_path, map_size=map_size)
    ep_num = 1000
    num_frames_to_sample = 32
    
    for episode_idx in range(ep_num):
        total_frames = np.random.randint(200,400)
        sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
        
        tokens=np.random.rand(num_frames_to_sample,1) #shape=(16, 2, 1374, 2048)
        
        with env.begin(write=True) as txn:
            for i in range(num_frames_to_sample):
                frame_index= sampled_indices[i]
                key_bytes = pickle.dumps((episode_idx,frame_index))
                tensor_bytes = pickle.dumps(tokens[i])
                compressed = zlib.compress(tensor_bytes, level=1)
                txn.put(key_bytes, compressed)
        if episode_idx%100==0:
            print(f"Loaded vggt tokens for episode {episode_idx}")
    env.close()
    
 
    print(f"数据库创建完成: {db_path}")
    

def load_lmdb_database(db_path):
    """加载LMDB数据库"""
    env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, meminit=False)
    return env

def reorganize_lmdb(db_path, map_size = 1099511627776):
    env = lmdb.open(db_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        keys_to_remap = []
        for key, compressed_value in txn.cursor():
            if 700 <= pickle.loads(key)[0] <900:
                txn.delete(key)
            elif pickle.loads(key)[0] >=900:
                value = zlib.decompress(compressed_value)
                value = pickle.loads(value)
                keys_to_remap.append((key, value))
                if pickle.loads(key)==(900,0):
                    print(value)
        frame_list = []
       
        for key, value in keys_to_remap:
            ep, frame = pickle.loads(key)

            new_ep = ep - 200
            new_key = pickle.dumps((new_ep, frame))
            txn.delete(key)
            if ep==900:
                frame_list.append(frame)
            compressed = pickle.dumps(value)
            compressed = zlib.compress(compressed, level=1)
            txn.put(new_key, compressed)
        print(len(frame_list))
    
    env.close()
    print("原地重组完成！")
    

def reorganize_lmdb_direct(db_path, map_size = 1099511627776):
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
            if ep == 900 and frame == 0:
                decompressed = zlib.decompress(value)
                decompressed = pickle.loads(decompressed)
                print(decompressed)
            new_ep = ep - 200
            new_key = pickle.dumps((new_ep, frame))
            txn.delete(key)
            txn.put(new_key, value)
    
    env.close()
    print("原地重组完成！")
    
def reorganize_lmdb_safe_fix(db_path):
    """
    安全修复版本 - 使用临时列表避免cursor冲突
    """
    env = lmdb.open(db_path, map_size=1099511627776, readonly=False)
    
    # 第一步：收集所有需要处理的信息
    all_entries = []
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            ep, frame = pickle.loads(key)
            all_entries.append({
                'key': key,
                'value': value,
                'ep': ep,
                'frame': frame
            })
    
    print(f"总条目数: {len(all_entries)}")
    
    # 第二步：执行删除和重映射
    with env.begin(write=True) as txn:
        delete_count = 0
        remap_count = 0
        keep_count = 0
        
        for entry in all_entries:
            key = entry['key']
            value = entry['value']
            ep = entry['ep']
            frame = entry['frame']
            
            if 700 <= ep <= 899:
                # 删除700-899
                if txn.delete(key):
                    delete_count += 1
                else:
                    print(f"删除失败: ep={ep}, frame={frame}")
                    
            elif ep >= 900:
                # 重映射900-999到700-799
                new_ep = ep - 200
                new_key = pickle.dumps((new_ep, frame))
                compressed = pickle.dumps(value)
                compressed = zlib.compress(compressed, level=1)
                # 先删除原key
                if txn.delete(key):
                    # 再插入新key
                    if txn.put(new_key, compressed):
                        remap_count += 1
                    else:
                        print(f"插入新key失败: {new_ep}, {frame}")
                else:
                    print(f"删除原key失败: ep={ep}, frame={frame}")
                    
            else:
                # 0-699保持不变
                keep_count += 1
    
    env.close()
    
    print(f"\n=== 操作统计 ===")
    print(f"保留条目 (0-699): {keep_count}")
    print(f"删除条目 (700-899): {delete_count}")
    print(f"重映射条目 (900-999→700-799): {remap_count}")
    print(f"总计: {keep_count + delete_count + remap_count}")
    

def verify_reorganization(db_path):
    env = lmdb.open(db_path, readonly=True, lock=False)
    
    episodes = set()
    total = set()
    
    with env.begin() as txn:
        frame_list = []
        for key, value in txn.cursor():
            ep, frame = pickle.loads(key)
            episodes.add(ep)
            # if ep < 700:
            total.add(pickle.loads(key))
            if ep==700:
                frame_list.append(frame)
                if frame == 0:
                    print("----------------")
                    value = zlib.decompress(value)
                    value = pickle.loads(value)
                    print(value)
        print(len(frame_list))
        print(len(total))
    
    env.close()
    
    actual_episodes = sorted(episodes)
    print(f"重组后的episode范围: {min(actual_episodes)} - {max(actual_episodes)}")
    
    # 检查是否包含不应该有的episode
    invalid_episodes = set(actual_episodes) & set(range(800, 1000))
    if invalid_episodes:
        print(f"❌ 错误：仍然包含800-999的数据: {sorted(invalid_episodes)}")
        return False
    
    # 检查700-799是否存在（由900-999映射而来）
    expected_lower = set(range(0, 700))
    expected_upper = set(range(700, 800))
    
    if expected_lower.issubset(episodes) and len(episodes & expected_upper) > 0:
        print("✅ 重组验证成功！")
        return True
    else:
        print("❌ 重组验证失败！")
        return False

if __name__ == "__main__":
    # create_lmdb_database_visual("./mydb")
    # verify_reorganization("./mydb")
    reorganize_lmdb_direct("./mydb")
    # verify_reorganization("./mydb")
    # env = load_lmdb_database("./mydb")
    # with env.begin() as txn:
    #     cursor = txn.cursor()
    #     print(len(list(cursor)))
    #     # for key_bytes, value_bytes in cursor:
    #     #     key = pickle.loads(key_bytes)
    #     #     value_bytes = txn.get(key_bytes)
    #     #     array = zlib.decompress(value_bytes)
    #     #     print(f"Key: {key}, Array shape: {array.shape}")
    #         # break  # 只打印第一个条目
    #     for key_bytes, _ in cursor:
    #         key = pickle.loads(key_bytes)
    #         value_bytes = txn.get(key_bytes)
    #         array = zlib.decompress(value_bytes)
    #         array = pickle.loads(array)

    #         print(f"Key: {key}, Array shape: {array.shape}")
    
    # dataset=BatchedLMDBDataset("/root/autodl-tmp/mydb")
    # enquirys = [(np.random.randint(0,1000),0) for i in range(20)]
    # # enquirys=[  (0,0),(0,1),(0,2),(0,3),(0,4),(0,7)]
    # tensors,tensor_masks=dataset[enquirys]
    # # print(enquirys,tensors,tensor_masks,tensors.shape)
    # print(enquirys,tensor_masks,tensors.shape)
    # print(tensors[0].dtype)
