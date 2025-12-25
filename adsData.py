import os
import sys

import moxing as mox
mox.file.shift('os','mox')
import io,gzip,base64,json
import re
import logging
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import cv2
import zstandard as zstd
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import random

import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Literal
from io import BytesIO
import base64
from collections.abc import Sequence
from collections.abc import Sequence as SequenceType

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder  # Commented out due to compatibility issues
import transformers

# from .uvp_module import data_list
from qwenvl.dataset.rope2d import get_rope_index_25, get_rope_index_2

# Import for action tokenizer
from transformers import AutoProcessor
import pickle
from qwenvl.utils.token_utils import prepare_action_tokenizer_mapping

import io
from ads.mods.connectors import ObsOs
import torch.distributed as dist
from .command_utils.behavior_features import calc_ego_behavior, EgoBehaviorType

import moxing 
import torch.nn.functional as F

from .tbt_extractor import (
    compute_tbt_sd_info_from_bag,
    merge_tbt_features,
    find_next_tbt,
    get_combo_tbt_from_pkl,
    get_lane_tbt_from_pkl,
)
from .tbt_formatter import get_tbt_text

moxing.file.shift('os','mox')

# from adsdata import uvp_module
# from adsdata import uvp_module
print(sys.path)
import uvp_module

os.environ["BUCKET_AREA"] = 'hnt2out-guiyang'

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

COMMAND_MAP = {
    EgoBehaviorType.KEEP:         "go straight",
    EgoBehaviorType.CHANGE_LEFT:  "go straight",
    EgoBehaviorType.CHANGE_RIGHT: "go straight",
    EgoBehaviorType.STRAIGHT:     "go straight",
    EgoBehaviorType.TURN_LEFT:    "go left",
    EgoBehaviorType.TURN_RIGHT:   "go right",
    EgoBehaviorType.UTURN:        "go left",
    EgoBehaviorType.UNKNOWN:      "unknown",
}

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor

def preprocess_qwen_2_visual_vla_sources(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List[int] = [],
    action_segments: Optional[List[torch.Tensor]] = None,
    action_roles: Optional[List[str]] = None,
    return_masks: bool = True,
) -> Dict:
    """Preprocess multiple sources (each a dialogue) with a single action segment list.

    - grid_thw_image: flattened list[int], one entry per <image> occurrence across all sources/turns
    - action_segments: list of LongTensor; consumed sequentially when a turn requires action insertion
      (user turns: historical actions → no loss; assistant turns: future actions → loss on action tokens)
    """
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_image_idx = 0
    action_idx = 0
    action_list: List[torch.Tensor] = [] if action_segments is None else (
        action_segments if isinstance(action_segments, list) else [action_segments]
    )
    action_role_list: List[str] = [] if action_roles is None else (
        action_roles if isinstance(action_roles, list) else [action_roles]
    )

    input_ids, targets = [], []
    
    img_token_id = vs_id = ve_id = None
    if return_masks:
        try:
            img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except Exception:
            img_token_id = vs_id = ve_id = None

    def expand_images(content: str) -> str:
        nonlocal visual_image_idx
        if "<image>" not in content:
            return content
        parts = content.split("<image>")
        new_parts = []
        for j in range(len(parts) - 1):
            new_parts.append(parts[j])
            replicate = grid_thw_image[visual_image_idx] if visual_image_idx < len(grid_thw_image) else 0
            replacement = (
                "<|vision_start|>" + ("<|image_pad|>" * replicate) + "<|vision_end|>"
            )
            new_parts.append(replacement)
            visual_image_idx += 1
        new_parts.append(parts[-1])
        return "".join(new_parts)
    
    for source in sources:
        cur_input, cur_target = [], []
        
        sys_tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."}
        ])
        cur_input += sys_tokens
        cur_target += [IGNORE_INDEX] * len(sys_tokens)

        norm_source = []
        for turn in source:
            if "role" in turn and "content" in turn:
                role = turn["role"]
                content = turn["content"]
            else:
                role = turn.get("from", turn.get("role"))
                content = turn.get("value", turn.get("content", ""))
            role = roles.get(role, role)
            norm_source.append({"role": role, "content": content})

        for turn in norm_source:
            role = turn["role"]
            content = expand_images(turn["content"])

            templ = tokenizer.apply_chat_template([{"role": role, "content": content}])

            inserted_action: List[int] = []
            should_insert = (
                action_idx < len(action_list)
                and (len(action_role_list) == 0 or (action_idx < len(action_role_list) and action_role_list[action_idx] == role))
            )
            if should_insert:
                inserted_action = action_list[action_idx].tolist()
                action_idx += 1

                insert_pos = len(templ) - 2
                enc = templ[:insert_pos] + inserted_action + templ[insert_pos:]
                lbl = [IGNORE_INDEX] * len(enc)
                if role == "assistant":
                    for k in range(3, insert_pos):
                        lbl[k] = enc[k]
                    for k in range(insert_pos, insert_pos + len(inserted_action)):
                        lbl[k] = enc[k]
                cur_input += enc
                cur_target += lbl
            else:
                enc = templ
                if role in ["user", "system"]:
                    lbl = [IGNORE_INDEX] * len(enc)
                else:
                    lbl = enc.copy()
                    lbl[:3] = [IGNORE_INDEX] * 3
                cur_input += enc
                cur_target += lbl

        input_ids.append(cur_input)
        targets.append(cur_target)

    out = dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(targets, dtype=torch.long),
    )
    
    if return_masks:
        input_tensor = torch.tensor(input_ids[0], dtype=torch.long) if input_ids else None
        
        if input_tensor is not None:
            seq_len = len(input_tensor)
            
            image_masks_list = []
            action_masks_list = []
            
            if img_token_id is not None and vs_id is not None and ve_id is not None:
                vs_positions = torch.where(input_tensor == vs_id)[0]
                ve_positions = torch.where(input_tensor == ve_id)[0]
                
                for vs_pos in vs_positions:
                    ve_candidates = ve_positions[ve_positions > vs_pos]
                    if len(ve_candidates) > 0:
                        ve_pos = ve_candidates[0]
                        
                        img_mask = torch.zeros(seq_len, dtype=torch.bool)
                        img_mask[vs_pos+1:ve_pos] = True
                        
                        image_masks_list.append([img_mask])
            
            boa_id = tokenizer.encode(tokenizer.boa_token, add_special_tokens=False)[0]
            eoa_id = tokenizer.encode(tokenizer.eoa_token, add_special_tokens=False)[0]

            
            if boa_id is not None and eoa_id is not None:
                boa_positions = torch.where(input_tensor == boa_id)[0]
                eoa_positions = torch.where(input_tensor == eoa_id)[0]
                
                for boa_pos in boa_positions:
                    eoa_candidates = eoa_positions[eoa_positions > boa_pos]
                    if len(eoa_candidates) > 0:
                        eoa_pos = eoa_candidates[0]
                        
                        act_mask = torch.zeros(seq_len, dtype=torch.bool)
                        if eoa_pos > boa_pos + 1:
                            act_mask[boa_pos+1:eoa_pos] = True
                        action_masks_list.append(act_mask)
            
            if image_masks_list:
                img_num = len(image_masks_list)
                T = 2
                n = math.ceil(img_num / T)
                image_masks_tensor = torch.zeros((T, n, seq_len), dtype=torch.bool)
                for t, masks in enumerate(image_masks_list):
                    if masks:
                        time_dim = t // n
                        num_dim = t % n
                        image_masks_tensor[time_dim, num_dim] = masks[0]
                out["image_token_masks"] = image_masks_tensor.unsqueeze(0)
            
            if action_masks_list:
                action_masks_tensor = torch.stack(action_masks_list, dim=0)
                out["action_future_masks"] = action_masks_tensor.unsqueeze(0)
    
    return out

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def smart_decoder(line):
    if line.strip().startswith('{'):
        return json.loads(line.strip())
    else:
        # 自动识别为被压缩的jsonl
        with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(line)), mode='rb') as f:
            line = pickle.load(f)
        return json.loads(line)

def remote_path(x):
    if not x.startswith('obs://'):
        # 如果不是s3的路径(obs://开通)，就说明是local path
        return x

    # bucket_area = BUCKET_AREA or self.use_beijing
    bucket_area = os.getenv("BUCKET_AREA")

    if bucket_area == "beijing":
        return re.sub('obs://.*?/','obs://ads-training-beijing/', x)
    elif bucket_area == 'wulan':
        if 'ads-ascend-battle-y-2' in x or 'ads-cloud-gy-y-2' in x:
            return re.sub('obs://.*?-2/','obs://ads-ascend-battle-y-2/', x)
        else:
            return re.sub('obs://.*?/','obs://ads-ascend-battle-y/', x)
    elif bucket_area == 'god':
        return x.replace('obs://god-training-data', 'obs://alluxio-131')
    elif bucket_area == 'suzhou':
        return re.sub('obs://.*?/','obs://god-training-data-sz/', x)
    elif bucket_area == "guiyang":
        x = x.replace("obs://god-training-data/data/god/autoscenes/","obs://god-training-data/data/god/autoscenes-prod/")
        if 'ads-ascend-battle-y-2' in x or 'ads-cloud-gy-y-2' in x:
            return re.sub('obs://.*?-2/','obs://ads-cloud-gy-y-2/', x)
        else:
            return re.sub('obs://.*?/','obs://ads-cloud-gy-y/', x)
    elif bucket_area == 'shanghai':
        return re.sub('obs://.*?/','obs://alluxio-131/', x)
    elif bucket_area == 'hnt2out-guiyang':
        return re.sub('obs://.*?/','obs://yw-ads-training-gy1/', x)

def txt(file_path, default=None):
    file_path = remote_path(file_path)
    try:
        with open(file_path, 'r') as f:
            return f.readlines()
    except:
        logging.error(f'file_path:"{file_path}" does not exist')
        return

def read_bytes(file_path, default=None, check_exist=False):
    file_path = remote_path(file_path)
    try:
        return mox.file.read_meta_free(file_path, binary=True)
    except:
        return default

def txt2jsonl(jsonl_files):
    jsonl_list = []
    for jsonl_file in jsonl_files:
        if jsonl_file.endswith('.txt'):
            lines = txt(jsonl_file)
            for line in lines:
                temp_line = line[:line.find('#')].strip() if line.find('#') != -1 else line.strip()
                if '.jsonl' in temp_line or '.txt' in temp_line or '.json' in temp_line:
                    jsonl_list.append(temp_line)
        else:
            jsonl_list.append(jsonl_file)
    return jsonl_list

def txt_lazy(file_path, default=None):
    file_path = remote_path(file_path)
    try:
        with mox.file.File(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                yield line
    except:
        print(f'file_path:"{file_path}" does not exist')

def match_re(match_str, jsonl_file):
    match = re.search(match_str, jsonl_file)
    replace_str, find_num = '', None
    if match:
        replace_str, find_num = match.group(0), match.group(1)
    return jsonl_file.replace(replace_str, ''), replace_str, find_num

def collate_fn(batch):
    batch_dict = dict()
    images = torch.stack([x[0] for x in batch])
    batch_dict.update({"images":images})
    pkl_keys = batch[0][1].keys()
    for key in pkl_keys:
        print(key)
        if key in ['curr_intersection_feat']:
            value = torch.stack([torch.tensor(instance[1][key].astype(np.float32)) for instance in batch]) # np.object
        else:
            value = torch.stack([torch.tensor(instance[1][key]) for instance in batch])
        batch_dict.update({key:value})
    
    return batch_dict

class ADSData(Dataset):
    def __init__(self, jsonl_files, cache_root, norm_json, suffix='LoadPlanning.LoadPlanningImpl.zstd.pkl'):
        self.jsonl_files = txt2jsonl(jsonl_files)
        self.cache_root = cache_root
        self.suffix = suffix
        self.data = self.get_image_and_pkl_set()
        self.rng = random.Random(42)
        self.norm = json.load(open(norm_json, 'r'))
        self.action_low = np.array(self.norm['norm_stats']['libero']['q01'], np.float32)
        self.action_high = np.array(self.norm['norm_stats']['libero']['q99'], np.float32)

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            raise ValueError("dist未初始化")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        try:
            self.clip_data = self.data[index]
            min = 2 + 3
            max = len(self.clip_data) - 1
            clip_index = self.rng.randint(min, max)
            img_path, pkl_path, command_pkl_path = self.clip_data[clip_index]
            img_tensor = self.get_img(img_path, 1)
            pkl_data = self.get_pkl(pkl_path)
            command_data = self.load_pickle(command_pkl_path)
            if img_tensor is None or pkl_data is None or command_data is None:
                raise ValueError("img or pkl is empty!")
            return img_tensor, pkl_data
        except Exception as err:
            # import traceback
            # print(traceback.print_exc())
            # print(f"err: {err}")
            index = np.random.choice(self.__len__())
            return self.__getitem__(index)
    
    def get_img(self, img_path, cam_id=1):
        image_path = remote_path(img_path)
        cam_column = f"cam_{cam_id}"
        image_path = image_path.replace("cam_1", cam_column)
        with io.BytesIO(mox.file.read_meta_free(image_path, binary=True)) as f:
            file_data = f.read()
            if image_path.endswith('parquet'):
                table = pq.read_table(pa.py_buffer(file_data)).to_pandas()
                if table is None:
                    return None
                data = table[cam_column].values[0]
            else:
                # 读取图像，默认读取为BGR格式
                data = file_data
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f'Image:{image_path} is None!')
            img = Image.fromarray(img[:, :, ::-1])
        return img


    def get_pkl(self, pkl_path):
        remote_cache = pkl_path.startswith("obs://")
        if not remote_cache or not os.path.exists(pkl_path):
            return None
        try:
            dctx = zstd.ZstdDecompressor()
            if remote_cache:
                compressed_data = read_bytes(pkl_path)
            else:
                with open(pkl_path, 'rb') as f:
                    compressed_data = f.read()
            val = pickle.loads(dctx.decompress(compressed_data))
            # 取需要字段，并进行预处理
            return (val[1], val[0])
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print(f"err:{e}")
            return None

    def get_image_and_pkl_set(self):
        data_set = []
        # count = 0
        for jsonl_file in self.jsonl_files:
            # if count >= 20000:
            #     break
            jsonl_file, _, duplicate = match_re(r'@duplicate(\d+)', jsonl_file)
            for idx, clip in enumerate(txt_lazy(jsonl_file)):
                clip_set = []
                clip = smart_decoder(clip)
                scene_id = clip['scene_id']
                scene_len = len(clip['samples'])
                if scene_len <= 5:
                    continue
                for frame in clip['samples']:
                    # count += 1
                    img_path = frame['img_cam1']
                    token = frame['sample_token']
                    pkl_path = os.path.join(self.cache_root, scene_id, token[0], token[1], token, self.suffix)
                    command_pkl = frame['planning_pkl']
                    # if not os.path.exists(pkl_path) or not os.path.exists(command_pkl):
                    #     continue
                    clip_set.append((img_path, pkl_path, command_pkl))
                data_set.append(clip_set)
        return data_set
    
    def rotate(self, vecs: np.ndarray, 
               rotate_angles: np.ndarray, 
               vec_mag_precalc: Optional[np.ndarray] = None,
                vec_angles_precalc: Optional[np.ndarray] = None):
        vec_magitudes = vec_mag_precalc if (vec_mag_precalc is not None) else np.linalg.norm(vecs, axis=-1)
        vec_angles = vec_angles_precalc if (vec_angles_precalc is not None) else np.arctan2(vecs[...,1], vecs[...,0])
        new_vec_angles = vec_angles + rotate_angles
        new_vecs = np.stack([vec_magitudes * np.cos(new_vec_angles), vec_magitudes * np.sin(new_vec_angles)]).T
        return new_vecs

    def make_segs_from_traj(self, traj: np.ndarray, with_heading=True):
        dxys = np.concatenate([traj[0:1,:2], (traj[1:,:2] - traj[:-1,:2])])  # [L,2]
        heads = traj[:,2] if with_heading else np.arctan2(dxys[:,1], dxys[:,0])  # [L]
        rot_dxys = np.concatenate([dxys[0:1,:], self.rotate(dxys[1:], -heads[:-1])])  # [L,2]
        relative_heads = np.concatenate([heads[0:1], (heads[1:] - heads[:-1])])  # [L]
        return np.concatenate([rot_dxys, relative_heads[:,None]], axis=1) if with_heading else rot_dxys
    
    def norm_rel_traj_pre_and_future(self, pre_actions, fut_actions):
        rel_pre_actions = self.make_segs_from_traj(pre_actions)
        rel_fut_actions = self.make_segs_from_traj(fut_actions)
        actions_array = np.concatenate([rel_pre_actions, rel_fut_actions])
        norm_actions = 2 * (actions_array - self.action_low) / (self.action_high - self.action_low + 1e-8) - 1
        norm_actions = np.clip(norm_actions, -1, 1)
        return norm_actions 
    
    def load_pickle(self, file_path: str, open_fn=open, pkl_type: Optional[Literal["zst", "pkl"]] = None):
        pkl_type = pkl_type or file_path.rsplit(".", 1)[-1]  # type: ignore
        if (pkl_type != "zst") and (pkl_type != "pkl"):
            raise ValueError(f"Unrecognized method: '{pkl_type}'")

        file_path = remote_path(file_path)
        if pkl_type == "pkl":
            with open_fn(file_path, "rb") as fp:
                data = pickle.load(fp)
        else:
            fp = open_fn(file_path, 'rb')
            fp_zstd = zstd.open(fp, 'rb')
            data = pickle.load(fp_zstd)
            fp_zstd.close()
            fp.close()
        return data
    
    def get_combo_tbt(self, tbt_feature, tbt_lane_feature):
        # 解析TBT信息
        key_tbt_indices = find_next_tbt(tbt_feature)
        combo_tbt = get_combo_tbt_from_pkl(tbt_feature, key_tbt_indices, len(tbt_feature))
        if tbt_lane_feature is None or len(tbt_lane_feature) <= 0:
            return combo_tbt

        for key, tbt_index in key_tbt_indices.items():
            if tbt_index < 0:
                continue

            (
                total_lanes,
                lane_ids,
                road_angles,
                lane_actions,
                optimal_lane_ids,
                extend_lanes,
            ) = get_lane_tbt_from_pkl(tbt_lane_feature[tbt_index])
            combo_tbt[key + "_total_lanes"] = total_lanes

            if len(road_angles) > 0:
                # 第一个out_road是导航选择的out_road（from @胡志强）
                _, combo_tbt[key + "_road_angle"] = road_angles[0]

            if len(lane_ids) <= 0:
                continue

            combo_tbt[key + "_lane"] = lane_ids
            combo_tbt[key + "_optimal_lane"] = optimal_lane_ids.tolist()
            combo_tbt[key + "_extend_lane"] = extend_lanes
            if len(lane_ids) != len(lane_actions):
                raise ValueError("Number of lane actions is not equal to number of lanes!")

            from collections import defaultdict
            lane_action_types_to_lanes = defaultdict(set)
            for i, lane_action in enumerate(lane_actions):
                lane_action_types_to_lanes[int(lane_action)].add(int(lane_ids[i]))
            combo_tbt[key + "_lane_action"] = lane_action_types_to_lanes

        return combo_tbt

    def build_tbt_features(self, lane_feat, tbt_feature):
        tbt_feature_list = [tuple(row) for row in tbt_feature.tolist()]

        tbt_num, B_num, _ = lane_feat.shape
        tbt_lane_feature_list = [
            np.full((B_num, 6), -1, dtype=lane_feat.dtype)
            for _ in range(tbt_num)
        ]

        merge_tbt_features(tbt_feature_list, tbt_lane_feature_list)
        combo_tbt = self.get_combo_tbt(tbt_feature_list, tbt_lane_feature_list)
        return get_tbt_text(combo_tbt, zh=True)[1]
    

class LazySupervisedHuawei2VAROSSDataset(ADSData):
    """NuPlan VLA Dataset building a single sequence from pre(1s) and cur.

    Differences from NavSim2:
    - No explicit pre_1s fields; compute pre image/action via 1s offset on the same sequence
    - Exactly one pre+cur sequence per sample (not two repeated cur anchors)
    - Time windows:
        pre action: 1.5s history, 1s future
        cur action: 1.5s history, 4s future
    - Index sampling controlled by rng(seed)
    """

    def __init__(self,
                tokenizer: transformers.PreTrainedTokenizer, 
                data_args):
        super().__init__(jsonl_files=[data_args.data_path], cache_root=data_args.cache_root, norm_json=data_args.norm_json)
        
        # Check VLA tokens
        if not (hasattr(tokenizer, 'boa_token') and hasattr(tokenizer, 'eoa_token')):
            raise ValueError("Tokenizer missing BOA/EOA tokens. Call check_and_add_vla_tokens first.")

        # Initialize Action Tokenizer
        if getattr(data_args, "actions_format", "fast") == "fast":
            self.fast_path = getattr(data_args, "action_tokenizer_path", None)
            if self.fast_path:
                self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)
            else:
                raise ValueError("action_tokenizer_path is required for fast actions format")
        else:
            raise ValueError(f"Unsupported actions_format: {getattr(data_args, 'actions_format', 'fast')}")

        # Action processing config
        self.use_actions = getattr(data_args, "use_actions", True)
        self.actions_format = getattr(data_args, "actions_format", "fast")

        # Get special token IDs
        mapping = prepare_action_tokenizer_mapping(tokenizer)
        self.boa_token_id = mapping["boa_token_id"]
        self.eoa_token_id = mapping["eoa_token_id"]
        self.last_vocab_idx = mapping["last_vocab_idx"]
        self.last_text_token_idx = mapping["last_vocab_idx"] - self.action_tokenizer.vocab_size
        rank0_print(f"VLA tokens: BOA={self.boa_token_id}, EOA={self.eoa_token_id}, last_vocab_idx={self.last_vocab_idx}")

        # Config
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        # Driving-specific config
        self.use_previous_actions = getattr(data_args, "use_previous_actions", False)
        self.action_hz = getattr(data_args, "action_hz", 2)
        self.frames_per_second = getattr(data_args, "frames_per_second", 2)  
        self.va_pair_num = 2  # fixed per requirement
        self.gap_frames = getattr(data_args, "va_gap_frames", self.frames_per_second)  # 1s gap default
        self.rng = random.Random(data_args.seed if hasattr(data_args, 'seed') else 42)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # ROSS features
        self.return_raw_vae = getattr(data_args, "return_raw_vae", True)
        self.return_masks = getattr(data_args, "return_masks", True)

        rank0_print("Huawei2VA: frames_per_second=%d, action_hz=%d, gap_frames=%d" % (self.frames_per_second, self.action_hz, self.gap_frames))

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        # Rough estimate: 2 images + two action segments (pre 1s fut + cur 4s fut)
        length_list = []
        for sample in self.data:
            img_tokens = 2 * 128 if "image" in sample else 0
            action_tokens = (int(1.0 * self.action_hz) + int(4.0 * self.action_hz))
            try:
                text_tokens = 200 if "text" in sample else 50
                length_list.append(text_tokens + img_tokens + action_tokens)
            except:
                length_list.append(400)
        return length_list

    # # ---------- Image helpers ----------
    # def _resolve_image_path(self, image_path):
    #     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    #     if (isinstance(image_path, str)
    #         and image_path.lower().endswith(image_extensions)
    #         and os.path.exists(image_path)):
    #         return image_path
    #     return self._construct_image_path(image_path)

    # def _construct_image_path(self, original_path):
    #     path_parts = original_path.split('/') if isinstance(original_path, str) else []
    #     if len(path_parts) >= 2:
    #         filename = path_parts[-1]
    #         if filename.endswith('.npy'):
    #             filename = filename[:-4]
    #         last_two = os.path.join(path_parts[-2], "CAM_F0", filename)
    #     else:
    #         filename = path_parts[-1] if path_parts else str(original_path)
    #         if isinstance(filename, str) and filename.endswith('.npy'):
    #             filename = filename[:-4]
    #         last_two = filename
    #     data_root = self.data_args.data_root
    #     actual_path = os.path.join(data_root, last_two + '.jpg')
    #     if os.path.exists(actual_path):
    #         return actual_path
    #     for ext in ['.jpeg', '.png']:
    #         alt_path = os.path.join(data_root, last_two + ext)
    #         if os.path.exists(alt_path):
    #             return alt_path
    #     print(f"Warning: Cannot resolve image path: {original_path}, trying original path")
    #     return original_path

    def process_image_unified(self, image_path, cam_id):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = self.get_img(image_path, cam_id)
        if image is None:
            raise ValueError("image is empty")
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if self.return_raw_vae:
            try:
                raw_vae_img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [3, H, W], [-1, 1]
            except Exception:
                raw_vae_img_tensor = None
            return image_tensor, grid_thw, raw_vae_img_tensor
        return image_tensor, grid_thw

    # ---------- BEV 可视化 helpers (最简实现) ----------
    def _convert_object_feat_to_obj_label(self, object_feat):
        """
        将 ADS object_feat 转换为 obj_label 格式 (N, 11)
        特征顺序: 0:x, 1:y, 2:heading, 3:class, 4:length, 5:width, 7:vel, 8:vel_orien
        输出: [x, y, z, lx, ly, lz, heading, category, state, vx, vy]
        """
        if object_feat is None:
            return None

        feat = np.array(object_feat)
        
        # shape: (N, T, feat_dim) -> 取当前帧
        # 注意: T维度通常是 [历史...当前...未来], 最后几帧可能是0(padding)
        # 根据数据结构, 当前帧通常在中间位置, 尝试多种策略找到有效帧
        if feat.ndim == 3:
            T = feat.shape[1]
            # 策略1: 尝试取中间帧 (当前帧通常在这里)
            mid_idx = T // 2
            # 策略2: 如果中间帧也是0, 尝试找第一个非零帧
            candidate_indices = [mid_idx, 0, T-1]  # 优先中间, 然后开头, 最后结尾
            
            for idx in candidate_indices:
                test_feat = feat[:, idx, :]
                # 检查是否有非零数据 (x,y坐标不全为0)
                if np.any(np.abs(test_feat[:, 0]) > 1e-6) or np.any(np.abs(test_feat[:, 1]) > 1e-6):
                    feat = test_feat
                    break
            else:
                # 所有候选都是0, 返回None
                return None
        
        if feat.shape[-1] < 9:
            return None

        N = feat.shape[0]
        x = feat[:, 0]
        y = feat[:, 1]
        heading = feat[:, 2]
        cls = feat[:, 3]
        length = feat[:, 4]
        width = feat[:, 5]
        speed = feat[:, 7]
        vel_orien = feat[:, 8]

        vx = speed * np.cos(vel_orien)
        vy = speed * np.sin(vel_orien)

        # 过滤无效数据 (padding)
        valid = ~((np.abs(x) < 1e-6) & (np.abs(y) < 1e-6))

        obj_label = np.zeros((N, 11), dtype=np.float32)
        obj_label[:, 0] = x
        obj_label[:, 1] = y
        obj_label[:, 2] = 0.0       # z
        obj_label[:, 3] = length
        obj_label[:, 4] = width
        obj_label[:, 5] = 1.5       # height default
        obj_label[:, 6] = heading
        obj_label[:, 7] = cls
        obj_label[:, 8] = 0.0       # state
        obj_label[:, 9] = vx
        obj_label[:, 10] = vy

        return obj_label[valid]

    def visualize_bev(self, cur_pkl_data, sample_id="debug"):
        """
        可视化 BEV semantic map (调试用)
        """
        if self.debug_bev_count >= self.debug_bev_max:
            return None  # 达到上限，跳过
            
        from huawei_code.transfuser_gt_utils_ads import (
            compute_bev_semantic_map_ads,
            compute_agent_targets_ads,
            visualize_bev_semantic_map,
        )
        
        # 1. 转换 object_feat -> obj_label
        object_feat = cur_pkl_data.get('object_feat', None)
        obj_label = self._convert_object_feat_to_obj_label(object_feat)
        
        if obj_label is None or len(obj_label) == 0:
            return None
        
        # 2. 获取静态物体 (车道线等)
        static_obj_feat = cur_pkl_data.get('static_obj_feat', None)
        static_obj_mask = cur_pkl_data.get('static_obj_mask', None)
        
        if static_obj_feat is not None:
            static_obj_feat = np.array(static_obj_feat)
        if static_obj_mask is not None:
            static_obj_mask = np.array(static_obj_mask)
        
        # 3. 生成 BEV semantic map
        bev_map = compute_bev_semantic_map_ads(
            obj_labels=obj_label,
            config=self.gt_config,
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
        )
        
        # 4. 生成 Agent targets
        agent_states, agent_labels = compute_agent_targets_ads(obj_label, self.gt_config)
        
        # 5. 可视化并保存
        save_path = os.path.join(self.debug_save_dir, f"bev_{sample_id}.png")
        visualize_bev_semantic_map(
            bev_map=bev_map,
            agent_states=agent_states,
            agent_labels=agent_labels,
            config=self.gt_config,
            save_path=save_path,
            title=f"Sample: {sample_id}",
        )
        
        self.debug_bev_count += 1
        rank0_print(f"[BEV Debug] Saved {save_path}, objects: {len(obj_label)}")
        return save_path

    # ---------- Action helpers ----------
    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.boa_token_id] + action_ids + [self.eoa_token_id], dtype=torch.long)

    def process_actions(self, actions: np.ndarray) -> List[int]:
        if self.actions_format == "fast":
            if isinstance(actions, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in actions]
                action_tokens = torch.cat(tensor_list, dim=0)
            else:
                action_tokens = torch.tensor(actions) if isinstance(actions, np.ndarray) else actions
            action_ids = self.action_tokenizer(action_tokens)[0]
            mapped_action_ids = [self.last_vocab_idx - id for id in action_ids]
            return mapped_action_ids
        else:
            raise ValueError(f"Unsupported actions_format: {self.actions_format}")

    def _sample_actions_window(self, actions: np.ndarray, hist_index: int = 3) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if actions is None :
            return None, None

        hist_tensor = None
        fut_wrapped = None

        if self.use_previous_actions:
            hist = actions[:hist_index]
            hist_ids = self.process_actions(hist)
            hist_tensor = torch.tensor(hist_ids, dtype=torch.long)

        fut = actions[hist_index:]
        fut_ids = self.process_actions(fut)
        fut_wrapped = self.wrap_action_sequence(fut_ids)

        return hist_tensor, fut_wrapped

    # ---------- Builders ----------
    def _grid_merge_count(self, grid_thw):
        merged = copy.deepcopy(grid_thw)
        if not isinstance(merged, SequenceType):
            merged = [merged]
        return [m.prod() // self.data_args.image_processor.merge_size**2 for m in merged]

    def _build_conversation_for_pairs(self, pairs_prompts: List[Tuple[str, str]]):
        # Each pair: (pre_prompt, cur_prompt)
        conv = []
        for pre_p, cur_p in pairs_prompts:
            conv.extend([
                {"from": "user", "value": f"<image>{pre_p}"},
                {"from": "assistant", "value": ""},
                {"from": "user", "value": f"<image>{cur_p}"},
                {"from": "assistant", "value": ""},
            ])
        return [conv]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(i)
        except Exception as e:
            i = np.random.choice(len(self.data))
            return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        scene = self.data[i]

        # Determine scene length: prefer action length if available, else fallback
        scene_len = len(scene)

        # Compute index constraints
        one_sec = int(round(self.frames_per_second * 1.0))
        step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
        hist_pre = int(round(1.5 * self.action_hz))
        # fut_pre = int(round(1.0 * self.action_hz))
        # hist_cur = int(round(1.5 * self.action_hz))
        # fut_cur = int(round(4.0 * self.action_hz))

        start_idx = one_sec + step * hist_pre
        # Ensure future windows fit (anchor + step*(count-1) within scene)
        # end_idx_pre = (scene_len - 1) - (step * (fut_pre - 1)) + one_sec
        end_idx = scene_len - 1
        # end_idx = min(end_idx_pre, end_idx_cur)

        low = max(0, start_idx)
        high = max(low, end_idx)
        if high < low:
            # degenerate window; fallback to center
            low = max(0, scene_len // 4)
            high = max(low, scene_len // 2)
        cur_idx = self.rng.randint(low, high)
        pre_idx = cur_idx - one_sec

        cur_img_path, cur_pkl_path, cur_command_pkl_path = scene[cur_idx]
        pre_img_path, pre_pkl_path, pre_command_pkl_path = scene[pre_idx]
        _, _15_cur_pkl_path, _ = scene[(cur_idx-3)]
        _, _15_pre_pkl_path, _ = scene[(pre_idx-3)]

        cur_pkl_data = self.get_pkl(cur_pkl_path)
        cur_command_data = self.load_pickle(cur_command_pkl_path, open_fn=mox.file.File)
        _15_cur_data = self.get_pkl(_15_cur_pkl_path)
        pre_pkl_data = self.get_pkl(pre_pkl_path)
        pre_command_data = self.load_pickle(pre_command_pkl_path, open_fn=mox.file.File)
        _15_pre_data = self.get_pkl(_15_pre_pkl_path)


        if cur_pkl_data is None or pre_pkl_data is None or cur_command_data is None or pre_command_data is None or _15_cur_data is None or _15_pre_data is None:
            print("cur_pkl_path: "+str(cur_pkl_path))
            print("cur_command_pkl_path: "+str(cur_command_pkl_path))
            print("_15_cur_pkl_path: "+str(_15_cur_pkl_path))
            print("pre_pkl_path: "+str(pre_pkl_path))
            print("pre_command_pkl_path: "+str(pre_command_pkl_path))
            print("_15_pre_pkl_path: "+str(_15_pre_pkl_path))
            raise ValueError("pkl is empty!")

        # Prepare containers
        all_images: List[torch.Tensor] = []
        all_image_thw: List[torch.Tensor] = []
        grid_thw_image_flat: List[int] = []
        action_segments: List[torch.Tensor] = []
        action_roles: List[str] = []
        # actions_array = scene.get("action", None)
        cur_action_fut = cur_pkl_data['full_gt_traj'][0,:8,:3] #8个waypoint
        cur_action_pre = _15_cur_data['full_gt_traj'][0,:3,:3] #3个waypoint
        pre_action_fut = pre_pkl_data['full_gt_traj'][0,:2,:3] #2个waypoint
        pre_action_pre = _15_pre_data['full_gt_traj'][0,:3,:3] #3个waypoint

        pre_actions_array = self.norm_rel_traj_pre_and_future(pre_action_pre, pre_action_fut)
        actions_array = self.norm_rel_traj_pre_and_future(cur_action_pre, cur_action_fut)

        cur_prompt_raw = calc_ego_behavior(cur_command_data)
        cur_prompt = COMMAND_MAP[cur_prompt_raw]
        pre_prompt_raw = calc_ego_behavior(pre_command_data)
        pre_prompt = COMMAND_MAP[pre_prompt_raw]

        # Images (pre, cur) with optional raw VAE support
        pre_image_raw = cur_image_raw = None
        pre_result = self.process_image_unified(pre_img_path, cam_id=1)
        cur_result = self.process_image_unified(cur_img_path, cam_id=1)

        if self.return_raw_vae:
            if pre_result is not None and len(pre_result) == 3:
                pre_img, pre_thw, pre_image_raw = pre_result
            else:
                pre_img, pre_thw = pre_result if pre_result else (None, None)
            if cur_result is not None and len(cur_result) == 3:
                cur_img, cur_thw, cur_image_raw = cur_result
            else:
                cur_img, cur_thw = cur_result if cur_result else (None, None)
        else:
            pre_img, pre_thw = pre_result if pre_result else (None, None)
            cur_img, cur_thw = cur_result if cur_result else (None, None)

        if pre_img is not None and pre_thw is not None:
            all_images.append(pre_img)
            all_image_thw.append(pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(pre_thw))
        if cur_img is not None and cur_thw is not None:
            all_images.append(cur_img)
            all_image_thw.append(cur_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(cur_thw))

        # Actions: pre then cur
        if self.use_actions and actions_array is not None:
            pre_hist, pre_fut = self._sample_actions_window(pre_actions_array, hist_index=3)
            if self.use_previous_actions and pre_hist is not None:
                action_segments.append(pre_hist)
                action_roles.append("user")
            if pre_fut is not None:
                action_segments.append(pre_fut)
                action_roles.append("assistant")

            cur_hist, cur_fut = self._sample_actions_window(actions_array, hist_index=3)
            if self.use_previous_actions and cur_hist is not None:
                action_segments.append(cur_hist)
                action_roles.append("user")
            if cur_fut is not None:
                action_segments.append(cur_fut)
                action_roles.append("assistant")

        # Build a single conversation with one pre+cur pair
        sources = self._build_conversation_for_pairs([(pre_prompt, cur_prompt)])

        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=sources,
            tokenizer=self.tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
            return_masks=self.return_masks,
        )

        # Position ids
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(all_image_thw, dim=0) if len(all_image_thw) > 0 else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if len(all_images) > 0:
            data_dict["pixel_values"] = torch.cat(all_images, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in all_image_thw], dim=0)

            if self.return_raw_vae:
                T, N = 2, 1
                if cur_image_raw is not None:
                    raw_shape = cur_image_raw.shape
                    raw_tensor = torch.zeros((T, N) + raw_shape, dtype=data_dict["pixel_values"].dtype)
                    if pre_image_raw is not None:
                        raw_tensor[0, 0] = pre_image_raw
                    raw_tensor[1, 0] = cur_image_raw
                    data_dict["raw_pixel_values_vae"] = raw_tensor
                    data_dict["frame_image_counts"] = torch.tensor([1, 1], dtype=torch.long)

        return data_dict
    

class LazySupervisedHuawei2VAROSSDataset_Multiview4(ADSData):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__(jsonl_files=[data_args.data_path], cache_root=data_args.cache_root, norm_json=data_args.norm_json)
        
        # Check VLA tokens
        if not (hasattr(tokenizer, 'boa_token') and hasattr(tokenizer, 'eoa_token')):
            raise ValueError("Tokenizer missing BOA/EOA tokens. Call check_and_add_vla_tokens first.")

        # Initialize Action Tokenizer
        if getattr(data_args, "actions_format", "fast") == "fast":
            self.fast_path = getattr(data_args, "action_tokenizer_path", None)
            if self.fast_path:
                self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)
            else:
                raise ValueError("action_tokenizer_path is required for fast actions format")
        else:
            raise ValueError(f"Unsupported actions_format: {getattr(data_args, 'actions_format', 'fast')}")

        # Action processing config
        self.use_actions = getattr(data_args, "use_actions", True)
        self.actions_format = getattr(data_args, "actions_format", "fast")

        # Get special token IDs
        mapping = prepare_action_tokenizer_mapping(tokenizer)
        self.boa_token_id = mapping["boa_token_id"]
        self.eoa_token_id = mapping["eoa_token_id"]
        self.last_vocab_idx = mapping["last_vocab_idx"]
        self.last_text_token_idx = mapping["last_vocab_idx"] - self.action_tokenizer.vocab_size
        rank0_print(f"VLA tokens: BOA={self.boa_token_id}, EOA={self.eoa_token_id}, last_vocab_idx={self.last_vocab_idx}")

        # Config
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        # Driving-specific config
        self.use_previous_actions = getattr(data_args, "use_previous_actions", False)
        self.action_hz = getattr(data_args, "action_hz", 2)
        self.frames_per_second = getattr(data_args, "frames_per_second", 2)  
        self.va_pair_num = 2  # fixed per requirement
        self.gap_frames = getattr(data_args, "va_gap_frames", self.frames_per_second)  # 1s gap default
        self.rng = random.Random(data_args.seed if hasattr(data_args, 'seed') else 42)

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # if not getattr(self.data_args, 'data_root', None):
        #     raise ValueError("data_root is required for Huawei VLA dataset")

        # ROSS features
        self.return_raw_vae = getattr(data_args, "return_raw_vae", True)
        self.return_masks = getattr(data_args, "return_masks", True)

        rank0_print("Huawei2VA: frames_per_second=%d, action_hz=%d, gap_frames=%d" % (self.frames_per_second, self.action_hz, self.gap_frames))

        # ========== BEV 可视化配置 (最简实现) ==========
        self.debug_bev = getattr(data_args, "debug_bev", False)
        self.debug_save_dir = getattr(data_args, "debug_save_dir", "/tmp/bev_debug")
        self.debug_bev_count = 0  # 控制可视化数量
        self.debug_bev_max = getattr(data_args, "debug_bev_max", 50)  # 最多保存50张
        
        if self.debug_bev:
            from huawei_code.transfuser_gt_utils_ads import ADSGTConfig
            self.gt_config = ADSGTConfig(
                vis_scale=8
            )
            os.makedirs(self.debug_save_dir, exist_ok=True)
            rank0_print(f"[BEV Debug] Enabled, saving to {self.debug_save_dir}")

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        # Rough estimate: 2 images + two action segments (pre 1s fut + cur 4s fut)
        length_list = []
        for sample in self.data:
            img_tokens = 2 * 128 if "image" in sample else 0
            action_tokens = (int(1.0 * self.action_hz) + int(4.0 * self.action_hz))
            try:
                text_tokens = 200 if "text" in sample else 50
                length_list.append(text_tokens + img_tokens + action_tokens)
            except:
                length_list.append(400)
        return length_list

    def process_image_unified(self, image_path, cam_id):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = self.get_img(image_path, cam_id)
        if image is None:
            raise ValueError("image is empty")
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if self.return_raw_vae:
            try:
                raw_vae_img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [3, H, W], [-1, 1]
            except Exception:
                raw_vae_img_tensor = None
            return image_tensor, grid_thw, raw_vae_img_tensor
        return image_tensor, grid_thw

    # ---------- BEV 可视化 helpers (最简实现) ----------
    def _convert_object_feat_to_obj_label(self, object_feat):
        """
        将 ADS object_feat 转换为 obj_label 格式 (N, 11)
        特征顺序: 0:x, 1:y, 2:heading, 3:class, 4:length, 5:width, 7:vel, 8:vel_orien
        输出: [x, y, z, lx, ly, lz, heading, category, state, vx, vy]
        """
        if object_feat is None:
            return None

        feat = np.array(object_feat)
        
        # shape: (N, T, feat_dim) -> 取最后一帧
        if feat.ndim == 3:
            feat = feat[:, -1, :]
        
        if feat.shape[-1] < 9:
            return None

        N = feat.shape[0]
        x = feat[:, 0]
        y = feat[:, 1]
        heading = feat[:, 2]
        cls = feat[:, 3]
        length = feat[:, 4]
        width = feat[:, 5]
        speed = feat[:, 7]
        vel_orien = feat[:, 8]

        vx = speed * np.cos(vel_orien)
        vy = speed * np.sin(vel_orien)

        # 过滤无效数据 (padding)
        valid = ~((np.abs(x) < 1e-6) & (np.abs(y) < 1e-6))

        obj_label = np.zeros((N, 11), dtype=np.float32)
        obj_label[:, 0] = x
        obj_label[:, 1] = y
        obj_label[:, 2] = 0.0       # z
        obj_label[:, 3] = length
        obj_label[:, 4] = width
        obj_label[:, 5] = 1.5       # height default
        obj_label[:, 6] = heading
        obj_label[:, 7] = cls
        obj_label[:, 8] = 0.0       # state
        obj_label[:, 9] = vx
        obj_label[:, 10] = vy

        return obj_label[valid]

    def visualize_bev(self, cur_pkl_data, sample_id="debug"):
        """
        可视化 BEV semantic map (调试用)
        """
        if self.debug_bev_count >= self.debug_bev_max:
            return None  # 达到上限，跳过
            
        from huawei_code.transfuser_gt_utils_ads import (
            compute_bev_semantic_map_ads,
            compute_agent_targets_ads,
            visualize_bev_semantic_map,
        )
        
        # 1. 转换 object_feat -> obj_label
        object_feat = cur_pkl_data.get('object_feat', None)
        obj_label = self._convert_object_feat_to_obj_label(object_feat)
        
        if obj_label is None or len(obj_label) == 0:
            return None
        
        # 2. 获取静态物体 (车道线等)
        static_obj_feat = cur_pkl_data.get('static_obj_feat', None)
        static_obj_mask = cur_pkl_data.get('static_obj_mask', None)
        
        if static_obj_feat is not None:
            static_obj_feat = np.array(static_obj_feat)
        if static_obj_mask is not None:
            static_obj_mask = np.array(static_obj_mask)
        
        # 3. 生成 BEV semantic map
        bev_map = compute_bev_semantic_map_ads(
            obj_labels=obj_label,
            config=self.gt_config,
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
        )
        
        # 4. 生成 Agent targets
        agent_states, agent_labels = compute_agent_targets_ads(obj_label, self.gt_config)
        
        # 5. 可视化并保存
        save_path = os.path.join(self.debug_save_dir, f"bev_{sample_id}.png")
        visualize_bev_semantic_map(
            bev_map=bev_map,
            agent_states=agent_states,
            agent_labels=agent_labels,
            config=self.gt_config,
            save_path=save_path,
            title=f"Sample: {sample_id}",
        )
        
        self.debug_bev_count += 1
        rank0_print(f"[BEV Debug] Saved {save_path}, objects: {len(obj_label)}")
        return save_path

    # ---------- Action helpers ----------
    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.boa_token_id] + action_ids + [self.eoa_token_id], dtype=torch.long)

    def process_actions(self, actions: np.ndarray) -> List[int]:
        if self.actions_format == "fast":
            if isinstance(actions, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in actions]
                action_tokens = torch.cat(tensor_list, dim=0)
            else:
                action_tokens = torch.tensor(actions) if isinstance(actions, np.ndarray) else actions
            action_ids = self.action_tokenizer(action_tokens)[0]
            mapped_action_ids = [self.last_vocab_idx - id for id in action_ids]
            return mapped_action_ids
        else:
            raise ValueError(f"Unsupported actions_format: {self.actions_format}")

    def _sample_actions_window(self, actions: np.ndarray, hist_index: int = 3) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if actions is None :
            return None, None

        hist_tensor = None
        fut_wrapped = None

        if self.use_previous_actions:
            hist = actions[:hist_index]
            hist_ids = self.process_actions(hist)
            hist_tensor = torch.tensor(hist_ids, dtype=torch.long)

        fut = actions[hist_index:]
        fut_ids = self.process_actions(fut)
        fut_wrapped = self.wrap_action_sequence(fut_ids)

        return hist_tensor, fut_wrapped

    # ---------- Builders ----------
    def _grid_merge_count(self, grid_thw):
        merged = copy.deepcopy(grid_thw)
        if not isinstance(merged, SequenceType):
            merged = [merged]
        return [m.prod() // self.data_args.image_processor.merge_size**2 for m in merged]

    def _build_conversation_for_pairs(self, pairs_prompts: List[Tuple[str, str]]):
        # Each pair: (pre_prompt, cur_prompt)
        conv = []
        for pre_p, cur_p in pairs_prompts:
            conv.extend([
                {"from": "user", "value": f"<image><image><image><image>{pre_p}"},
                {"from": "assistant", "value": ""},
                {"from": "user", "value": f"<image><image><image><image>{cur_p}"},
                {"from": "assistant", "value": ""},
            ])
        return [conv]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(i)
        except Exception as e:
            i = np.random.choice(len(self.data))
            return self.__getitem__(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        scene = self.data[i]

        # Determine scene length: prefer action length if available, else fallback
        scene_len = len(scene)

        # Compute index constraints
        one_sec = int(round(self.frames_per_second * 1.0))
        step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
        hist_pre = int(round(1.5 * self.action_hz))

        start_idx = one_sec + step * hist_pre
        end_idx = scene_len - 1

        low = max(0, start_idx)
        high = max(low, end_idx)
        if high < low:
            # degenerate window; fallback to center
            # low = max(0, scene_len // 4)
            # high = max(low, scene_len // 2)
            low = 5
            high = end_idx
        cur_idxs = list(range(low, high))
        self.rng.shuffle(cur_idxs)
        idx = 0
        try:
            return self.get_item(cur_idxs[idx], scene_len, scene)
        except Exception as e:
            print(f"data invalid:({i},{cur_idxs[idx]})")
            idx += 1
            if idx > len(cur_idxs) - 1:
                raise ValueError("Current Clip has not valid data!") 
            return self.get_item(cur_idxs[idx], scene_len, scene)

    def get_item(self, cur_idx, scene_len, scene):
        pre_idx = cur_idx - 2
        if scene_len <= cur_idx:
            raise ValueError("Clip is too short!")  
        cur_img_path, cur_pkl_path, cur_command_pkl_path = scene[cur_idx]
        pre_img_path, pre_pkl_path, pre_command_pkl_path = scene[pre_idx]
        _, _15_cur_pkl_path, _ = scene[(cur_idx-3)]
        _, _15_pre_pkl_path, _ = scene[(pre_idx-3)]

        cur_pkl_data, extra_cur_pkl_data = self.get_pkl(cur_pkl_path)
        cur_command_data = self.load_pickle(cur_command_pkl_path, open_fn=mox.file.File)
        _15_cur_data, _ = self.get_pkl(_15_cur_pkl_path)
        pre_pkl_data, extra_pre_pkl_data = self.get_pkl(pre_pkl_path)
        pre_command_data = self.load_pickle(pre_command_pkl_path, open_fn=mox.file.File)
        _15_pre_data, _ = self.get_pkl(_15_pre_pkl_path)


        if cur_pkl_data is None or pre_pkl_data is None or cur_command_data is None or pre_command_data is None or _15_cur_data is None or _15_pre_data is None:
            raise ValueError("pkl is empty!")     

        # Prepare containers
        all_images: List[torch.Tensor] = []
        all_image_thw: List[torch.Tensor] = []
        grid_thw_image_flat: List[int] = []
        action_segments: List[torch.Tensor] = []
        action_roles: List[str] = []
        # actions_array = scene.get("action", None)
        cur_action_fut = cur_pkl_data['full_gt_traj'][0,:10,:3] #10个waypoint
        cur_action_pre = _15_cur_data['full_gt_traj'][0,:3,:3] #3个waypoint
        pre_action_fut = pre_pkl_data['full_gt_traj'][0,:2,:3] #2个waypoint
        pre_action_pre = _15_pre_data['full_gt_traj'][0,:3,:3] #3个waypoint

        pre_actions_array = self.norm_rel_traj_pre_and_future(pre_action_pre, pre_action_fut)
        actions_array = self.norm_rel_traj_pre_and_future(cur_action_pre, cur_action_fut)

        # cur_prompt_raw = calc_ego_behavior(cur_command_data)
        # cur_prompt = COMMAND_MAP[cur_prompt_raw]
        # pre_prompt_raw = calc_ego_behavior(pre_command_data)
        # pre_prompt = COMMAND_MAP[pre_prompt_raw]
        cur_tbt_prompt = self.build_tbt_features(cur_command_data["tbt_lane_feature_from_bag"][0], cur_command_data["tbt_feature_from_bag"][0])
        pre_tbt_prompt = self.build_tbt_features(pre_command_data["tbt_lane_feature_from_bag"][0], pre_command_data["tbt_feature_from_bag"][0])

        # [BEV Debug] 可视化调用
        if getattr(self, 'debug_bev', False) and cur_pkl_data is not None:
            self.visualize_bev(cur_pkl_data, sample_id=f"{cur_idx}")

        # Images (pre, cur) with optional raw VAE support
        pre_image_raw = cur_image_raw = None
        left_pre_image_raw = left_image_raw = None
        right_pre_image_raw = right_image_raw = None
        back_pre_image_raw = back_image_raw = None

        pre_result = self.process_image_unified(pre_img_path, cam_id=1)
        left_pre_result = self.process_image_unified(pre_img_path, cam_id=5)
        right_pre_result = self.process_image_unified(pre_img_path, cam_id=6)
        back_pre_result = self.process_image_unified(pre_img_path, cam_id=8)

        cur_result = self.process_image_unified(cur_img_path, cam_id=1)
        left_result = self.process_image_unified(cur_img_path, cam_id=5)
        right_result = self.process_image_unified(cur_img_path, cam_id=6)
        back_result = self.process_image_unified(cur_img_path, cam_id=8)

        if self.return_raw_vae:
            pre_img, pre_thw, pre_image_raw = pre_result
            left_pre_img, left_pre_thw, left_pre_image_raw = left_pre_result
            right_pre_img, right_pre_thw, right_pre_image_raw = right_pre_result
            back_pre_img, back_pre_thw, back_pre_image_raw = back_pre_result

            cur_img, cur_thw, cur_image_raw = cur_result
            left_img, left_thw, left_image_raw = left_result
            right_img, right_thw, right_image_raw = right_result
            back_img, back_thw, back_image_raw = back_result
        else:
            pre_img, pre_thw = pre_result
            left_pre_img, left_pre_thw = left_pre_result
            right_pre_img, right_pre_thw = right_pre_result
            back_pre_img, back_pre_thw = back_pre_result

            cur_img, cur_thw = cur_result
            left_img, left_thw = left_result
            right_img, right_thw = right_result
            back_img, back_thw = back_result

        if pre_img is not None and pre_thw is not None:
            all_images.append(left_pre_img)
            all_image_thw.append(left_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(left_pre_thw))
        
            all_images.append(pre_img)
            all_image_thw.append(pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(pre_thw))

            all_images.append(right_pre_img)
            all_image_thw.append(right_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(right_pre_thw))

            all_images.append(back_pre_img)
            all_image_thw.append(back_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(back_pre_thw))
        if cur_img is not None and cur_thw is not None:
            all_images.append(left_img)
            all_image_thw.append(left_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(left_thw))

            all_images.append(cur_img)
            all_image_thw.append(cur_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(cur_thw))

            all_images.append(right_img)
            all_image_thw.append(right_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(right_thw))

            all_images.append(back_img)
            all_image_thw.append(back_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(back_thw))

        # Actions: pre then cur
        if self.use_actions and actions_array is not None:
            pre_hist, pre_fut = self._sample_actions_window(pre_actions_array, hist_index=3)
            if self.use_previous_actions and pre_hist is not None:
                action_segments.append(pre_hist)
                action_roles.append("user")
            if pre_fut is not None:
                action_segments.append(pre_fut)
                action_roles.append("assistant")

            cur_hist, cur_fut = self._sample_actions_window(actions_array, hist_index=3)
            if self.use_previous_actions and cur_hist is not None:
                action_segments.append(cur_hist)
                action_roles.append("user")
            if cur_fut is not None:
                action_segments.append(cur_fut)
                action_roles.append("assistant")

        # Build a single conversation with one pre+cur pair
        sources = self._build_conversation_for_pairs([(pre_tbt_prompt, cur_tbt_prompt)])

        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=sources,
            tokenizer=self.tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
            return_masks=self.return_masks,
        )

        # Position ids
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(all_image_thw, dim=0) if len(all_image_thw) > 0 else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        # data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if len(all_images) > 0:
            data_dict["pixel_values"] = torch.cat(all_images, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in all_image_thw], dim=0)

            if self.return_raw_vae:
                T, N = 2, 4
                raw_shape = cur_image_raw.shape
                raw_tensor = torch.zeros((T, N) + raw_shape, dtype=data_dict["pixel_values"].dtype)
                raw_tensor[0, 0] = left_pre_image_raw
                raw_tensor[0, 1] = pre_image_raw
                raw_tensor[0, 2] = right_pre_image_raw
                raw_tensor[0, 3] = back_pre_image_raw

                raw_tensor[1, 0] = left_image_raw
                raw_tensor[1, 1] = cur_image_raw
                raw_tensor[1, 2] = right_image_raw
                raw_tensor[1, 3] = back_image_raw

                data_dict["raw_pixel_values_vae"] = raw_tensor
                data_dict["frame_image_counts"] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
                data_dict["gt_action"] = actions_array

        return data_dict
    

class LazySupervisedHuawei2VAROSSMOEDataset_Multiview4(ADSData):
    def __init__(self,
                tokenizer: transformers.PreTrainedTokenizer, 
                data_args):
        super().__init__(jsonl_files=[data_args.data_path], cache_root=data_args.cache_root, norm_json=data_args.norm_json)
        
        # Action processing config
        ObsOs.AutoConfig()
        ObsOs.SetCacheDir(f"/cache/rank{dist.get_rank() if dist.is_initialized() else 0}")
        # Check VLA tokens
        if not (hasattr(tokenizer, 'boa_token') and hasattr(tokenizer, 'eoa_token')):
            raise ValueError("Tokenizer missing BOA/EOA tokens. Call check_and_add_vla_tokens first.")

        # Initialize Action Tokenizer
        if getattr(data_args, "actions_format", "fast") == "fast":
            self.fast_path = getattr(data_args, "action_tokenizer_path", None)
            if self.fast_path:
                self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)
            else:
                raise ValueError("action_tokenizer_path is required for fast actions format")
        else:
            raise ValueError(f"Unsupported actions_format: {getattr(data_args, 'actions_format', 'fast')}")

        # Action processing config
        self.use_actions = getattr(data_args, "use_actions", True)
        self.actions_format = getattr(data_args, "actions_format", "fast")

        # Get special token IDs
        mapping = prepare_action_tokenizer_mapping(tokenizer)
        self.boa_token_id = mapping["boa_token_id"]
        self.eoa_token_id = mapping["eoa_token_id"]
        self.last_vocab_idx = mapping["last_vocab_idx"]
        self.last_text_token_idx = mapping["last_vocab_idx"] - self.action_tokenizer.vocab_size

        rank0_print(f"VLA tokens: BOA={self.boa_token_id}, EOA={self.eoa_token_id}, last_vocab_idx={self.last_vocab_idx}")

        # Config
        self.model_type = data_args.model_type
        if data_args.model_type.startswith("qwen2.5vl"):
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        # Driving-specific config
        self.use_previous_actions = getattr(data_args, "use_previous_actions", True)
        self.action_hz = getattr(data_args, "action_hz", 2)
        self.frames_per_second = getattr(data_args, "frames_per_second", 2)  
        self.va_pair_num = 2  # fixed per requirement
        self.gap_frames = getattr(data_args, "va_gap_frames", self.frames_per_second)  # 1s gap default
        self.rng = random.Random(data_args.seed if hasattr(data_args, 'seed') else 42)
        self.cur_idx = data_args.cur_frame_idx
        self.training = data_args.training

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # if not getattr(self.data_args, 'data_root', None):
        #     raise ValueError("data_root is required for Huawei VLA dataset")

        # ROSS features
        self.return_raw_vae = getattr(data_args, "return_raw_vae", True)
        self.return_masks = getattr(data_args, "return_masks", True)

        # cmd for action expert
        self.text_name_list = [
            "go left",
            "go straight",
            "go right",
            "unknown",
        ]
        self.prompt2vec = {
            name: F.one_hot(torch.tensor(i), num_classes=len(self.text_name_list)).float()
            for i, name in enumerate(self.text_name_list)
        }

        rank0_print("Huawei2VA: frames_per_second=%d, action_hz=%d, gap_frames=%d" % (self.frames_per_second, self.action_hz, self.gap_frames))

        # ========== BEV 可视化配置 (最简实现) ==========
        self.debug_bev = getattr(data_args, "debug_bev", False)
        self.debug_save_dir = getattr(data_args, "debug_save_dir", "/tmp/bev_debug")
        self.debug_bev_count = 0  # 控制可视化数量
        self.debug_bev_max = getattr(data_args, "debug_bev_max", 50)  # 最多保存50张
        
        if self.debug_bev:
            from huawei_code.transfuser_gt_utils_ads import ADSGTConfig
            self.gt_config = ADSGTConfig(
                bev_pixel_width=256,
                bev_pixel_height=128,
                lidar_min_x=0.0,
                lidar_max_x=32.0,
                lidar_min_y=-32.0,
                lidar_max_y=32.0,
                use_fov_filter=False,  # 多视角不需要FOV过滤
            )
            os.makedirs(self.debug_save_dir, exist_ok=True)
            rank0_print(f"[BEV Debug] Enabled, saving to {self.debug_save_dir}")

    def __len__(self):
        return len(self.data)
    
    @property
    def lengths(self):
        # Rough estimate: 2 images + two action segments (pre 1s fut + cur 4s fut)
        length_list = []
        for sample in self.data:
            img_tokens = 2 * 128 if "image" in sample else 0
            action_tokens = (int(1.0 * self.action_hz) + int(4.0 * self.action_hz))
            try:
                text_tokens = 200 if "text" in sample else 50
                length_list.append(text_tokens + img_tokens + action_tokens)
            except:
                length_list.append(400)
        return length_list

    def process_image_unified(self, image_path, cam_id):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = self.get_img(image_path, cam_id)
        if image is None:
            raise ValueError("image is empty")
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        if self.return_raw_vae:
            try:
                raw_vae_img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # [3, H, W], [-1, 1]
            except Exception:
                raw_vae_img_tensor = None
            return image_tensor, grid_thw, raw_vae_img_tensor
        return image_tensor, grid_thw

    # ---------- BEV 可视化 helpers (最简实现) ----------
    def _convert_object_feat_to_obj_label(self, object_feat):
        """
        将 ADS object_feat 转换为 obj_label 格式 (N, 11)
        特征顺序: 0:x, 1:y, 2:heading, 3:class, 4:length, 5:width, 7:vel, 8:vel_orien
        输出: [x, y, z, lx, ly, lz, heading, category, state, vx, vy]
        """
        if object_feat is None:
            return None

        feat = np.array(object_feat)
        
        # shape: (N, T, feat_dim) -> 取最后一帧
        if feat.ndim == 3:
            feat = feat[:, -1, :]
        
        if feat.shape[-1] < 9:
            return None

        N = feat.shape[0]
        x = feat[:, 0]
        y = feat[:, 1]
        heading = feat[:, 2]
        cls = feat[:, 3]
        length = feat[:, 4]
        width = feat[:, 5]
        speed = feat[:, 7]
        vel_orien = feat[:, 8]

        vx = speed * np.cos(vel_orien)
        vy = speed * np.sin(vel_orien)

        # 过滤无效数据 (padding)
        valid = ~((np.abs(x) < 1e-6) & (np.abs(y) < 1e-6))

        obj_label = np.zeros((N, 11), dtype=np.float32)
        obj_label[:, 0] = x
        obj_label[:, 1] = y
        obj_label[:, 2] = 0.0       # z
        obj_label[:, 3] = length
        obj_label[:, 4] = width
        obj_label[:, 5] = 1.5       # height default
        obj_label[:, 6] = heading
        obj_label[:, 7] = cls
        obj_label[:, 8] = 0.0       # state
        obj_label[:, 9] = vx
        obj_label[:, 10] = vy

        return obj_label[valid]

    def visualize_bev(self, cur_pkl_data, sample_id="debug"):
        """
        可视化 BEV semantic map (调试用)
        """
        if self.debug_bev_count >= self.debug_bev_max:
            return None  # 达到上限，跳过
            
        from huawei_code.transfuser_gt_utils_ads import (
            compute_bev_semantic_map_ads,
            compute_agent_targets_ads,
            visualize_bev_semantic_map,
        )
        
        # 1. 转换 object_feat -> obj_label
        object_feat = cur_pkl_data.get('object_feat', None)
        obj_label = self._convert_object_feat_to_obj_label(object_feat)
        
        if obj_label is None or len(obj_label) == 0:
            return None
        
        # 2. 获取静态物体 (车道线等)
        static_obj_feat = cur_pkl_data.get('static_obj_feat', None)
        static_obj_mask = cur_pkl_data.get('static_obj_mask', None)
        
        if static_obj_feat is not None:
            static_obj_feat = np.array(static_obj_feat)
        if static_obj_mask is not None:
            static_obj_mask = np.array(static_obj_mask)
        
        # 3. 生成 BEV semantic map
        bev_map = compute_bev_semantic_map_ads(
            obj_labels=obj_label,
            config=self.gt_config,
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
        )
        
        # 4. 生成 Agent targets
        agent_states, agent_labels = compute_agent_targets_ads(obj_label, self.gt_config)
        
        # 5. 可视化并保存
        save_path = os.path.join(self.debug_save_dir, f"bev_{sample_id}.png")
        visualize_bev_semantic_map(
            bev_map=bev_map,
            agent_states=agent_states,
            agent_labels=agent_labels,
            config=self.gt_config,
            save_path=save_path,
            title=f"Sample: {sample_id}",
        )
        
        self.debug_bev_count += 1
        rank0_print(f"[BEV Debug] Saved {save_path}, objects: {len(obj_label)}")
        return save_path

    # ---------- Action helpers ----------
    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        return torch.tensor([self.boa_token_id] + action_ids + [self.eoa_token_id], dtype=torch.long)

    def process_actions(self, actions: np.ndarray) -> List[int]:
        if self.actions_format == "fast":
            if isinstance(actions, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in actions]
                action_tokens = torch.cat(tensor_list, dim=0)
            else:
                action_tokens = torch.tensor(actions) if isinstance(actions, np.ndarray) else actions
            action_ids = self.action_tokenizer(action_tokens)[0]
            mapped_action_ids = [self.last_vocab_idx - id for id in action_ids]
            return mapped_action_ids
        else:
            raise ValueError(f"Unsupported actions_format: {self.actions_format}")

    def _sample_actions_window(self, actions: np.ndarray, hist_index: int = 3) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if actions is None :
            return None, None

        hist_tensor = None
        fut_wrapped = None

        if self.use_previous_actions:
            hist = actions[:hist_index]
            hist_ids = self.process_actions(hist)
            hist_tensor = torch.tensor(hist_ids, dtype=torch.long)

        fut = actions[hist_index:]
        fut_ids = self.process_actions(fut)
        fut_wrapped = self.wrap_action_sequence(fut_ids)

        return hist_tensor, fut_wrapped

    # ---------- Builders ----------
    def _grid_merge_count(self, grid_thw):
        merged = copy.deepcopy(grid_thw)
        if not isinstance(merged, SequenceType):
            merged = [merged]
        return [m.prod() // self.data_args.image_processor.merge_size**2 for m in merged]

    def _build_conversation_for_pairs(self, pairs_prompts: List[Tuple[str, str]]):
        # Each pair: (pre_prompt, cur_prompt)
        conv = []
        for pre_p, cur_p in pairs_prompts:
            conv.extend([
                {"from": "user", "value": f"<image><image><image><image>{pre_p}"},
                {"from": "assistant", "value": ""},
                {"from": "user", "value": f"<image><image><image><image>{cur_p}"},
                {"from": "assistant", "value": ""},
            ])
        return [conv]
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(i)
        except Exception as e:
            i = np.random.choice(len(self.data))
            return self.__getitem__(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        scene = self.data[i]

        # Determine scene length: prefer action length if available, else fallback
        scene_len = len(scene)

        # Compute index constraints
        one_sec = int(round(self.frames_per_second * 1.0))
        step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
        hist_pre = int(round(1.5 * self.action_hz))

        start_idx = one_sec + step * hist_pre
        end_idx = scene_len - 1

        low = max(0, start_idx)
        high = max(low, end_idx)
        if high < low:
            # degenerate window; fallback to center
            # low = max(0, scene_len // 4)
            # high = max(low, scene_len // 2)
            low = 5
            high = end_idx
        cur_idxs = list(range(low, high))
        self.rng.shuffle(cur_idxs)
        idx = 0
        try:
            return self.get_item(cur_idxs[idx], scene_len, scene)
        except Exception as e:
            print(f"data invalid:({i},{cur_idxs[idx]})")
            idx += 1
            if idx > len(cur_idxs) - 1:
                raise ValueError("Current Clip has not valid data!") 
            return self.get_item(cur_idxs[idx], scene_len, scene)

    def get_item(self, cur_idx, scene_len, scene):
        pre_idx = cur_idx - 2
        if scene_len <= cur_idx:
            raise ValueError("Clip is too short!")  
        cur_img_path, cur_pkl_path, cur_command_pkl_path = scene[cur_idx]
        pre_img_path, pre_pkl_path, pre_command_pkl_path = scene[pre_idx]
        _, _15_cur_pkl_path, _ = scene[(cur_idx-3)]
        _, _15_pre_pkl_path, _ = scene[(pre_idx-3)]

        cur_pkl_data, extra_cur_pkl_data = self.get_pkl(cur_pkl_path)
        cur_command_data = self.load_pickle(cur_command_pkl_path, open_fn=mox.file.File)
        _15_cur_data, _ = self.get_pkl(_15_cur_pkl_path)
        pre_pkl_data, extra_pre_pkl_data = self.get_pkl(pre_pkl_path)
        pre_command_data = self.load_pickle(pre_command_pkl_path, open_fn=mox.file.File)
        _15_pre_data, _ = self.get_pkl(_15_pre_pkl_path)


        if cur_pkl_data is None or pre_pkl_data is None or cur_command_data is None or pre_command_data is None or _15_cur_data is None or _15_pre_data is None:
            raise ValueError("pkl is empty!")     

        # Prepare containers
        all_images: List[torch.Tensor] = []
        all_image_thw: List[torch.Tensor] = []
        grid_thw_image_flat: List[int] = []
        action_segments: List[torch.Tensor] = []
        action_roles: List[str] = []
        # actions_array = scene.get("action", None)
        cur_action_fut = cur_pkl_data['full_gt_traj'][0,:10,:3] #10个waypoint
        cur_action_pre = _15_cur_data['full_gt_traj'][0,:3,:3] #3个waypoint
        pre_action_fut = pre_pkl_data['full_gt_traj'][0,:2,:3] #2个waypoint
        pre_action_pre = _15_pre_data['full_gt_traj'][0,:3,:3] #3个waypoint

        pre_actions_array = self.norm_rel_traj_pre_and_future(pre_action_pre, pre_action_fut)
        actions_array = self.norm_rel_traj_pre_and_future(cur_action_pre, cur_action_fut)

        # cur_prompt_raw = calc_ego_behavior(cur_command_data)
        # cur_prompt = COMMAND_MAP[cur_prompt_raw]
        # pre_prompt_raw = calc_ego_behavior(pre_command_data)
        # pre_prompt = COMMAND_MAP[pre_prompt_raw]
        cur_tbt_prompt = self.build_tbt_features(cur_command_data["tbt_lane_feature_from_bag"][0], cur_command_data["tbt_feature_from_bag"][0])
        pre_tbt_prompt = self.build_tbt_features(pre_command_data["tbt_lane_feature_from_bag"][0], pre_command_data["tbt_feature_from_bag"][0])

        # [BEV Debug] 可视化调用
        if getattr(self, 'debug_bev', False) and cur_pkl_data is not None:
            self.visualize_bev(cur_pkl_data, sample_id=f"{cur_idx}")

        # Images (pre, cur) with optional raw VAE support
        pre_image_raw = cur_image_raw = None
        left_pre_image_raw = left_image_raw = None
        right_pre_image_raw = right_image_raw = None
        back_pre_image_raw = back_image_raw = None

        pre_result = self.process_image_unified(pre_img_path, cam_id=1)
        left_pre_result = self.process_image_unified(pre_img_path, cam_id=5)
        right_pre_result = self.process_image_unified(pre_img_path, cam_id=6)
        back_pre_result = self.process_image_unified(pre_img_path, cam_id=8)

        cur_result = self.process_image_unified(cur_img_path, cam_id=1)
        left_result = self.process_image_unified(cur_img_path, cam_id=5)
        right_result = self.process_image_unified(cur_img_path, cam_id=6)
        back_result = self.process_image_unified(cur_img_path, cam_id=8)

        if self.return_raw_vae:
            pre_img, pre_thw, pre_image_raw = pre_result
            left_pre_img, left_pre_thw, left_pre_image_raw = left_pre_result
            right_pre_img, right_pre_thw, right_pre_image_raw = right_pre_result
            back_pre_img, back_pre_thw, back_pre_image_raw = back_pre_result

            cur_img, cur_thw, cur_image_raw = cur_result
            left_img, left_thw, left_image_raw = left_result
            right_img, right_thw, right_image_raw = right_result
            back_img, back_thw, back_image_raw = back_result
        else:
            pre_img, pre_thw = pre_result
            left_pre_img, left_pre_thw = left_pre_result
            right_pre_img, right_pre_thw = right_pre_result
            back_pre_img, back_pre_thw = back_pre_result

            cur_img, cur_thw = cur_result
            left_img, left_thw = left_result
            right_img, right_thw = right_result
            back_img, back_thw = back_result

        if pre_img is not None and pre_thw is not None:
            all_images.append(left_pre_img)
            all_image_thw.append(left_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(left_pre_thw))
        
            all_images.append(pre_img)
            all_image_thw.append(pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(pre_thw))

            all_images.append(right_pre_img)
            all_image_thw.append(right_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(right_pre_thw))

            all_images.append(back_pre_img)
            all_image_thw.append(back_pre_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(back_pre_thw))
        if cur_img is not None and cur_thw is not None:
            all_images.append(left_img)
            all_image_thw.append(left_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(left_thw))

            all_images.append(cur_img)
            all_image_thw.append(cur_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(cur_thw))

            all_images.append(right_img)
            all_image_thw.append(right_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(right_thw))

            all_images.append(back_img)
            all_image_thw.append(back_thw)
            grid_thw_image_flat.extend(self._grid_merge_count(back_thw))

        # Actions: pre then cur
        if self.use_actions and actions_array is not None:
            pre_hist, pre_fut = self._sample_actions_window(pre_actions_array, hist_index=3)
            if self.use_previous_actions and pre_hist is not None:
                action_segments.append(pre_hist)
                action_roles.append("user")
            if pre_fut is not None:
                action_segments.append(pre_fut)
                action_roles.append("assistant")

            cur_hist, cur_fut = self._sample_actions_window(actions_array, hist_index=3)
            if self.use_previous_actions and cur_hist is not None:
                action_segments.append(cur_hist)
                action_roles.append("user")
            if cur_fut is not None:
                action_segments.append(cur_fut)
                action_roles.append("assistant")

        # Build a single conversation with one pre+cur pair
        sources = self._build_conversation_for_pairs([(pre_tbt_prompt, cur_tbt_prompt)])

        data_dict = preprocess_qwen_2_visual_vla_sources(
            sources=sources,
            tokenizer=self.tokenizer,
            grid_thw_image=grid_thw_image_flat,
            action_segments=action_segments if len(action_segments) > 0 else None,
            action_roles=action_roles if len(action_roles) > 0 else None,
            return_masks=self.return_masks,
        )

        action_ids_tensor = cur_fut
        # 固定长度 padding 到 24; 超长截断并确保以 <eoa> 结尾
        MAX_AR_LEN = 24
        pad_id = self.tokenizer.pad_token_id
        if action_ids_tensor.shape[0] >= MAX_AR_LEN:
            action_fixed = action_ids_tensor[:MAX_AR_LEN].clone()
            action_fixed[-1] = self.eoa_token_id  # 保证结尾为 <eoa>
            labels_fixed = action_fixed.clone()
        else:
            pad_len = MAX_AR_LEN - action_ids_tensor.shape[0]
            action_fixed = torch.cat([
                action_ids_tensor,
                torch.full((pad_len,), pad_id, dtype=action_ids_tensor.dtype)
            ], dim=0)
            labels_fixed = torch.cat([
                action_ids_tensor,
                torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)
            ], dim=0)
        
        data_dict["vlm_input_ids"] = data_dict["input_ids"]
        data_dict["vlm_labels"] = data_dict["labels"]

        data_dict["input_ids"] = action_fixed
        data_dict["action_input_ids"] = action_fixed     
        data_dict['labels'] = labels_fixed
    
        # Position ids
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["vlm_input_ids"],
            image_grid_thw=torch.stack(all_image_thw, dim=0) if len(all_image_thw) > 0 else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids

        if len(all_images) > 0:
            data_dict["pixel_values"] = torch.cat(all_images, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in all_image_thw], dim=0)

            if self.return_raw_vae:
                T, N = 2, 4
                raw_shape = cur_image_raw.shape
                raw_tensor = torch.zeros((T, N) + raw_shape, dtype=data_dict["pixel_values"].dtype)
                raw_tensor[0, 0] = left_pre_image_raw
                raw_tensor[0, 1] = pre_image_raw
                raw_tensor[0, 2] = right_pre_image_raw
                raw_tensor[0, 3] = back_pre_image_raw

                raw_tensor[1, 0] = left_image_raw
                raw_tensor[1, 1] = cur_image_raw
                raw_tensor[1, 2] = right_image_raw
                raw_tensor[1, 3] = back_image_raw
                data_dict["raw_pixel_values_vae"] = raw_tensor
                data_dict["frame_image_counts"] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)

        # 补充action expert所需要的输入
        # data_dict["cmd"] = self.prompt2vec[cur_tbt_prompt]
        data_dict["action"] = torch.tensor(np.array(actions_array)[3:, :], dtype=torch.float)
        data_dict["pre_action"] = torch.tensor(np.array(actions_array)[:3, :], dtype=torch.float)
        
        return data_dict
    


@dataclass
class DataCollatorForSupervisedVLADataset(object):
    """Collate examples for supervised VLA fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # Handle images/videos (same as original)
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0 and videos[0] != None:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        # Handle raw VAE pixel values and frame counts
        raw_pixel_values_vae = [instance["raw_pixel_values_vae"] for instance in instances if "raw_pixel_values_vae" in instance]
        frame_image_counts = [instance["frame_image_counts"] for instance in instances if "frame_image_counts" in instance]
        
        if len(raw_pixel_values_vae) != 0:
            # raw_pixel_values_vae shape: [T, N, ...] -> concat along N (batch) dimension
            # Since each instance has N=1, we concat them to create actual batch dimension
            batch["raw_pixel_values_vae"] = torch.cat(raw_pixel_values_vae, dim=1)  # concat along N dimension
        else:
            batch["raw_pixel_values_vae"] = None
            
        if len(frame_image_counts) != 0:
            # frame_image_counts shape: [T] -> stack to create batch dimension
            batch["frame_image_counts"] = torch.stack(frame_image_counts, dim=0)  # create batch dimension
        else:
            batch["frame_image_counts"] = None
        
        # Handle image token masks and action future masks
        image_token_masks = [instance["image_token_masks"] for instance in instances if "image_token_masks" in instance]
        action_future_masks = [instance["action_future_masks"] for instance in instances if "action_future_masks" in instance]
        
        if len(image_token_masks) != 0 and len(action_future_masks) != 0:
            # 将变长的掩码在时间维(最后一维)对齐到同一长度，并在 batch 维度拼接
            max_img_len = max(mask.shape[-1] for mask in image_token_masks)
            max_act_len = max(mask.shape[-1] for mask in action_future_masks)
            target_length = max(max_img_len, max_act_len)

            padded_image_masks = []
            padded_action_masks = []
            for img_mask, act_mask in zip(image_token_masks, action_future_masks):
                pad_img = target_length - img_mask.shape[-1]
                pad_act = target_length - act_mask.shape[-1]

                if pad_img > 0:
                    img_mask = torch.nn.functional.pad(img_mask, (0, pad_img), mode='constant', value=0)
                if pad_act > 0:
                    act_mask = torch.nn.functional.pad(act_mask, (0, pad_act), mode='constant', value=0)

                padded_image_masks.append(img_mask)
                padded_action_masks.append(act_mask)

            batch["image_token_masks"] = torch.cat(padded_image_masks, dim=0)
            batch["action_future_masks"] = torch.cat(padded_action_masks, dim=0)
        else:
            batch["image_token_masks"] = None if len(image_token_masks) == 0 else image_token_masks
            batch["action_future_masks"] = None if len(action_future_masks) == 0 else action_future_masks
               
        return batch
    

@dataclass
class DataCollatorForSupervisedVLAMoEDataset(object):
    """Collate examples for supervised VLA fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vlm_input_ids, vlm_labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("vlm_input_ids", "vlm_labels", "position_ids")
        )
        vlm_input_ids = [ids.squeeze(0) for ids in vlm_input_ids]
        vlm_labels = [ids.squeeze(0) for ids in vlm_labels]
        vlm_input_ids = torch.nn.utils.rnn.pad_sequence(
            vlm_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        vlm_labels = torch.nn.utils.rnn.pad_sequence(
            vlm_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        vlm_input_ids = vlm_input_ids[:, : self.tokenizer.model_max_length]
        vlm_labels = vlm_labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        
        batch = dict(
            vlm_input_ids=vlm_input_ids,
            vlm_labels=vlm_labels,
            vlm_attention_mask=vlm_input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # Handle images/videos (same as original)
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        # Handle raw VAE pixel values and frame counts
        raw_pixel_values_vae = [instance["raw_pixel_values_vae"] for instance in instances if "raw_pixel_values_vae" in instance]
        frame_image_counts = [instance["frame_image_counts"] for instance in instances if "frame_image_counts" in instance]
        
        if len(raw_pixel_values_vae) != 0:
            # raw_pixel_values_vae shape: [T, N, ...] -> concat along N (batch) dimension
            # Since each instance has N=1, we concat them to create actual batch dimension
            batch["raw_pixel_values_vae"] = torch.cat(raw_pixel_values_vae, dim=1)  # concat along N dimension
        else:
            batch["raw_pixel_values_vae"] = None
            
        if len(frame_image_counts) != 0:
            # frame_image_counts shape: [T] -> stack to create batch dimension
            batch["frame_image_counts"] = torch.stack(frame_image_counts, dim=0)  # create batch dimension
        else:
            batch["frame_image_counts"] = None
        
        # Handle image token masks and action future masks
        image_token_masks = [instance["image_token_masks"] for instance in instances if "image_token_masks" in instance]
        action_future_masks = [instance["action_future_masks"] for instance in instances if "action_future_masks" in instance]
        
        if len(image_token_masks) != 0 and len(action_future_masks) != 0:
            # 将变长的掩码在时间维(最后一维)对齐到同一长度，并在 batch 维度拼接
            max_img_len = max(mask.shape[-1] for mask in image_token_masks)
            max_act_len = max(mask.shape[-1] for mask in action_future_masks)
            target_length = max(max_img_len, max_act_len)

            padded_image_masks = []
            padded_action_masks = []
            for img_mask, act_mask in zip(image_token_masks, action_future_masks):
                pad_img = target_length - img_mask.shape[-1]
                pad_act = target_length - act_mask.shape[-1]

                if pad_img > 0:
                    img_mask = torch.nn.functional.pad(img_mask, (0, pad_img), mode='constant', value=0)
                if pad_act > 0:
                    act_mask = torch.nn.functional.pad(act_mask, (0, pad_act), mode='constant', value=0)

                padded_image_masks.append(img_mask)
                padded_action_masks.append(act_mask)

            batch["image_token_masks"] = torch.cat(padded_image_masks, dim=0)
            batch["action_future_masks"] = torch.cat(padded_action_masks, dim=0)
        else:
            batch["image_token_masks"] = None if len(image_token_masks) == 0 else image_token_masks
            batch["action_future_masks"] = None if len(action_future_masks) == 0 else action_future_masks


        # 添加 action expert 相关字段
        input_ids_stack = [instance["input_ids"] for instance in instances if "input_ids" in instance]
        labels_stack = [instance["labels"] for instance in instances if "labels" in instance]
        pre_action_stack = [instance["pre_action"] for instance in instances if "pre_action" in instance]
        action_stack = [instance["action"] for instance in instances if "action" in instance]
        cmd_stack = [instance["cmd"] for instance in instances if "cmd" in instance]

        if len(input_ids_stack) > 0:
            batch["input_ids"] = torch.stack(input_ids_stack, dim=0)
        if len(labels_stack) > 0:
            batch["labels"] = torch.stack(labels_stack, dim=0)        
        if len(pre_action_stack) > 0:
            batch["pre_action"] = torch.stack(pre_action_stack, dim=0)
        if len(action_stack) > 0:
            batch["action"] = torch.stack(action_stack, dim=0)
        if len(cmd_stack) > 0:
            batch["cmd"] = torch.stack(cmd_stack, dim=0)

        return batch


def make_supervised_data_module_huawei2_vla_ross(tokenizer: transformers.PreTrainedTokenizer, data_args, jsonl_files, cache_root, norm_json) -> Dict:
    """Make NuPlan 2VA dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedHuawei2VAROSSDataset(tokenizer=tokenizer, data_args=data_args, jsonl_files=jsonl_files, cache_root=cache_root, norm_json=norm_json)
    data_collator = DataCollatorForSupervisedVLADataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def make_supervised_data_module_huawei2_vla_ross_multiview4_5kw(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedHuawei2VAROSSDataset_Multiview4(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedVLADataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5kw(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make NuPlan 2VA dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedHuawei2VAROSSMOEDataset_Multiview4(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedVLAMoEDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
