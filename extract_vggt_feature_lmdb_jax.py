import os
import torch
import numpy as np
from tqdm import tqdm
import lmdb
import pickle
import zlib
from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp

# -------------------------------
# Import new model and related utility functions
# -------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# -------------------------------
# Device and data type settings
# -------------------------------
# If you have only one GPU, change "cuda:1" to "cuda:0"
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)
if torch.cuda.is_available():
    # Get the current GPU's compute capability: Ampere GPU (Compute Capability 8.0+) supports bfloat16
    dev_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32
print(dtype)

# -------------------------------
# Initialize the VGGT model and load pretrained weights
# -------------------------------

model = VGGT()
checkpoint = torch.load('checkpoints/model.pt', map_location=device)
msg = model.load_state_dict(checkpoint)
print('loading status:', msg)
model = model.to(device).eval()

# -------------------------------
# Data storage path settings
# -------------------------------
root_dir = '/root/autodl-tmp/rlbench_data_224/data_train_224'
dataset_list = os.listdir(root_dir)

num_frames_to_sample = 32 # 32
image_view = ['main', 'wrist']

# -------------------------------
# Create lmdb database
# -------------------------------
db_path = "/root/autodl-tmp/mydb"
# remove existing database if exists
if os.path.exists(db_path):
    print(f"Removing existing database at {db_path}")
    for file in os.listdir(db_path):
        os.remove(os.path.join(db_path, file))

map_size=300*1024*1024*1024
env = lmdb.open(db_path, map_size=map_size)

# -------------------------------
# Iterate over each episode and sample a fixed number of frames
# -------------------------------
episode_offset = 0
# for raw_dataset_name in RAW_DATASET_NAMES:
for raw_dataset_name in dataset_list:
    print(f"开始创建{raw_dataset_name}")
    dataset_dir = os.path.join(root_dir, raw_dataset_name)
    episode_list = os.listdir(dataset_dir)
    episode_num = len(episode_list)
    print(f"{raw_dataset_name}共{episode_num}个episode")
    # episode_list = episode_list[:1]

    for episode in tqdm(episode_list):
        # print(episode)
        episode_dir = os.path.join(dataset_dir, episode)
        if not os.path.isdir(episode_dir):
            continue

        # Ensure that each episode has a corresponding save directory
        # scene_save_dir = os.path.join(root_save_3d_feature, episode)
        # if not os.path.exists(scene_save_dir):
        #     os.makedirs(scene_save_dir)

        total_frames = len([file for file in os.listdir(episode_dir) if file.endswith('.jpg')])/2
        sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)

        total_features = []
        total_image_list = []

        for view in image_view:
            # Get all jpg images in the episode, sort them by filename, then sample a fixed number of frames
            file_names = [file for file in os.listdir(episode_dir) if file.endswith(f'{view}.jpg')]
            file_names.sort()
            file_total_frames = len(file_names)
            # print(total_frames)
            if file_total_frames == 0:
                continue
            sampled_file_list = [os.path.join(episode_dir, file_names[i]) for i in sampled_indices]
            total_image_list.append(sampled_file_list)

            # -------------------------------
            # Load images
            # -------------------------------
            # Here we use the load_and_preprocess_images function provided by vggt
            # The returned result is assumed to have shape (N, 3, H, W)
            images = load_and_preprocess_images(sampled_file_list)
            images = images.to(device, non_blocking=True)
            images = images.to(dtype)
            # Add batch dimension, final shape is (1, num_frames, 3, H, W)
            images = images.unsqueeze(0)

            # -------------------------------
            # Model inference, feature extraction
            # -------------------------------
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                    aggregated_tokens_list, ps_idx = model.aggregator(images)

            # -------------------------------
            # Save aggregated_tokens_list[-1] and ps_idx
            # -------------------------------
            # Move the output to CPU and convert to numpy format
            feature = aggregated_tokens_list[-1].to(dtype=torch.float16)
            # feature = torch.rand(1, 6, 1)
            total_features.append(feature[...,5:,:])
            # print(feature.dtype)
            # print(feature.shape)

            # Assume ps_idx is a tensor, otherwise save directly
            # ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx

        features = torch.cat(total_features, dim=0) #shape=(2, 16, 1369, 2048)
        features = features.transpose(1, 0) #shape=(16, 2, 1369, 2048)
        features = features.cpu().numpy()
        features = jnp.array(features, dtype=jnp.bfloat16)
        # print(features.dtype)
        # print(features.shape)
        
        # episode_idx = int(episode[2:])-1
        episode_idx = int(episode.split('_')[1])
        # print(episode_idx)
        with env.begin(write=True) as txn:
            for i, im_name in enumerate(total_image_list[0]):
                strings=im_name.split('/')
                frame_index = strings[-1].split('_')[2]
                frame_index=int(frame_index[4:])
                # frame_index=int(strings[-1][4:10])
                # key_bytes = pickle.dumps((episode_idx+episode_offset,frame_index-1))
                key_bytes = pickle.dumps((episode_idx+episode_offset,frame_index))
                tensor_bytes = pickle.dumps(features[i].reshape(-1,2048))
                # tensor_bytes = pickle.dumps(features[i].reshape(-1,1))
                compressed = zlib.compress(tensor_bytes, level=1)
                txn.put(key_bytes, compressed)

    episode_offset += episode_num

env.close()
print(f"数据库创建完成: {db_path}")
