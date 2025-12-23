import os
if os.getenv('USING_ASCEND_910B') == "1":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import moxing 
moxing.file.shift('os','mox')
import pickle
from ego_hmf_checker_predict import EgoHmfCheckerPredict
from scipy.interpolate import interp1d
import copy
import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from llm_utils import CaseVisualization, DynamicObject, images_to_video, tensor_traj_to_str
import shutil

def compute_averages(data, filter_key="is_valid_for_checker"):
    """
    从 list of dicts 中筛选出 filter_key 为 True 的 dict，
    然后对指定的 keys 进行平均。

    参数:
        data (list of dict): 输入的字典列表
        filter_key (str): 用于筛选的 key（只有当它为 True 时才参与）
        keys_to_average (list of str): 需要平均的 key 列表

    返回:
        dict: 每个 key 对应的平均值，例如 {'key1': 0.5, 'key2': 1.2, ...}
    """
    keys_to_average = ['ADE_30','FDE_30','Pred_Frame_Collision_Rate_1s','Pred_Frame_Collision_Rate_3s',\
        'Pred_Frame_Responsible_Collision_Rate_1s','Pred_Frame_Responsible_Collision_Rate_3s',\
        'Pred_Static_Collision_Rate_1s','Pred_Static_Collision_Rate_3s',\
        'Pred_Off_Road_Rate_1s','Pred_Off_Road_Rate_3s','Lateral_Goal_offset','On_Solid']

    sums = {key: 0.0 for key in keys_to_average}
    counts = {key: 0 for key in keys_to_average}

    for item in data:
        if bool(item.get(filter_key, False).item()):  # 只有当 filter_key 为 True 时才参与
            for key in keys_to_average:
                value = item.get(key)
                if value is not None:
                    sums[key] += value.item()
                    counts[key] += 1
                else:
                    print('has NAN::::')
                    print(item)

    # 计算平均值，防止除以 0
    average_dict = {
        key: (sums[key] / counts[key] if counts[key] > 0 else None)
        for key in keys_to_average
    }

    return average_dict

def plot_trajectories(original_traj_pred, interpolated_traj_pred, original_traj_gt, interpolated_traj_gt, original_times, interpolated_times, json_file_path):
    """
    绘制原始轨迹和插值后的轨迹对比图
    :param original_traj: 原始轨迹 tensor (N, 3)
    :param interpolated_traj: 插值后轨迹 tensor (M, 3)
    :param original_times: 原始时间戳 list 或 array
    :param interpolated_times: 插值后时间戳 list 或 array
    """
    original_traj_pred = original_traj_pred.numpy()
    interpolated_traj_pred = interpolated_traj_pred.numpy()
    original_traj_gt = original_traj_gt.numpy()
    interpolated_traj_gt = interpolated_traj_gt.numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(original_traj_pred[:, 0], original_traj_pred[:, 1], 'o-', label='Original traj pred')
    plt.plot(interpolated_traj_pred[:, 0], interpolated_traj_pred[:, 1], 'x-', label='interpolated traj pred')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('traj_pred')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(original_traj_gt[:, 0], original_traj_gt[:, 1], 'o-', label='Original traj gt')
    plt.plot(interpolated_traj_gt[:, 0], interpolated_traj_gt[:, 1], 'x-', label='interpolated traj gt')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('traj_gt')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('debug.png')
    plt.close()

def interpolate_traj_3d(traj_tensor, original_timestamps, target_timestamps):
    """
    插值轨迹的 x, y, heading 三个维度，支持外推
    :param traj_tensor: shape (N, 3)，原始轨迹 [x, y, heading]
    :param original_timestamps: list or np.array，原始时间戳
    :param target_timestamps: list or np.array，目标时间戳
    :return: 插值后的轨迹，shape (M, 3)
    """
    traj_np = traj_tensor.numpy()
    
    # 使用 fill_value="extrapolate" 允许外推
    fx = interp1d(original_timestamps, traj_np[:, 0], kind='linear', fill_value="extrapolate")
    fy = interp1d(original_timestamps, traj_np[:, 1], kind='linear', fill_value="extrapolate")
    # fheading = interp1d(original_timestamps, traj_np[:, 2], kind='linear', fill_value="extrapolate")

    # 插值
    new_x = fx(target_timestamps)
    new_y = fy(target_timestamps)
    # new_heading = fheading(target_timestamps)

    # 合并为 (M, 3)
    # new_traj = np.stack([new_x, new_y, new_heading], axis=-1)
    new_traj = np.stack([new_x, new_y], axis=-1)

    # 转回 tensor
    return torch.from_numpy(new_traj).float()
    
def rotate(vecs: np.ndarray, rotate_angles: np.ndarray, vec_mag_precalc: Optional[np.ndarray] = None,
           vec_angles_precalc: Optional[np.ndarray] = None):
    vec_magitudes = vec_mag_precalc if (vec_mag_precalc is not None) else np.linalg.norm(vecs, axis=-1)
    vec_angles = vec_angles_precalc if (vec_angles_precalc is not None) else np.arctan2(vecs[...,1], vecs[...,0])
    new_vec_angles = vec_angles + rotate_angles
    new_vecs = np.stack([vec_magitudes * np.cos(new_vec_angles), vec_magitudes * np.sin(new_vec_angles)]).T
    return new_vecs 
 
def make_traj_from_segs(segs: np.ndarray, x_idx: int, y_idx: int, theta_idx: Optional[int], append_origin=False):
    if segs.ndim != 2:
         raise ValueError
    dxys_raw = segs[:,[x_idx, y_idx]]
    if theta_idx is None:
        dthetas = np.arctan2(segs[:,y_idx], segs[:,x_idx])
        dxys_rotate = rotate(dxys_raw[1:], dthetas[:-1].cumsum(axis=0), vec_angles_precalc=dthetas[1:])
        dxys_rotate = np.concatenate([dxys_raw[:1], dxys_rotate])
    else:
        dthetas = segs[:,theta_idx]
        dxys_rotate = np.concatenate([dxys_raw[:1], rotate(dxys_raw[1:], dthetas[:-1].cumsum(axis=0))])
    xys = dxys_rotate.cumsum(axis=0)
    if append_origin:
        xys = np.concatenate([np.array([[0.0, 0.0]], dtype=xys.dtype), xys])
    return xys 
 
def make_segs_from_traj(traj: np.ndarray, with_heading=True):
    dxys = np.concatenate([traj[0:1,:2], (traj[1:,:2] - traj[:-1,:2])])  # [L,2]
    heads = traj[:,2] if with_heading else np.arctan2(dxys[:,1], dxys[:,0])  # [L]
    rot_dxys = np.concatenate([dxys[0:1,:], rotate(dxys[1:], -heads[:-1])])  # [L,2]
    relative_heads = np.concatenate([heads[0:1], (heads[1:] - heads[:-1])])  # [L]
    return np.concatenate([rot_dxys, relative_heads[:,None]], axis=1) if with_heading else rot_dxys

# 查找对应的 pkl 文件
def find_pkl_file(folder_path, scene_id, timestamp):
    prefix = f"{scene_id}_{timestamp}"
    for file in os.listdir(folder_path):
        if file.startswith(prefix) and file.endswith(".pkl"):
            return os.path.join(folder_path, file)
    return None

def calculate_errors(traj1, traj2, idx):
    """
    计算两条轨迹之间的距离误差
    traj1和traj2应该是形状相同的numpy数组
    """
    # 确保两条轨迹长度相同
    min_len = min(len(traj1), len(traj2))
    traj1 = traj1[:min_len]
    traj2 = traj2[:min_len]
    
    # 计算对应点之间的欧氏距离
    distances = np.linalg.norm(traj1 - traj2, axis=1)
    
    # 前六个点的平均距离误差 (ADE)
    ade = np.mean(distances[:idx]) if min_len >= 6 else np.mean(distances)
    
    # 第六个点的最终距离误差 (FDE)
    fde = distances[idx-1] if min_len >= 6 else distances[-1]
    
    return ade, fde

def vvis_trajectory(traj_pred_checker, inputs, images_folder):
    vis = CaseVisualization()
    n_hist = 1
    for j in range(len(inputs[0]['obj_label']) // n_hist):
        key_frame_index = (j + 1) * n_hist - 1
        scene_name = inputs[0]['sceneName'][key_frame_index]
        ts_token = inputs[0]['ts_token'][key_frame_index]
        imgs_paths = inputs[0]['imgs_paths'][key_frame_index]

        obj_label = inputs[0]['obj_label'][key_frame_index]
        obj_pred_tensor = torch.zeros((1, 100, 11)).to(obj_label.device)
        obj_pred = obj_pred_tensor[key_frame_index]
        OBJ_List = []
        for x in range(obj_label.shape[0]):
            value = obj_label[x].cpu().numpy()
            # if np.mean(value) == -1:
            if np.abs(np.mean(value[:3])+1)<1e-6:
                continue
            OBJ_List.append(DynamicObject(value,has_velocity=True))            
        OBJ_List_Pred = []
        for x in range(obj_pred.shape[0]):
            value = copy.deepcopy(obj_pred[x]).cpu().numpy()
            value[3]=np.exp(value[3])
            value[4]=np.exp(value[4])
            value[5]=np.exp(value[5])
            if np.abs(np.mean(value[:3])+1)<1e-6:
                continue
            OBJ_List_Pred.append(DynamicObject(value,has_velocity=True))
        # print('obj_label length {}, obj_pred length {}'.format(len(OBJ_List),len(OBJ_List_Pred)))

        static_obj = inputs[1]['static_obj_feat'] * inputs[1]['static_obj_mask'][..., -1].unsqueeze(-1).unsqueeze(-1)
        curb_obj = inputs[1]['curb_feat'] * inputs[1]['curb_mask'][..., -1].unsqueeze(-1).unsqueeze(-1)
        static_obj, curb_obj = static_obj[key_frame_index], curb_obj[key_frame_index]
        STATIC_OBJ_list = []
        for x in range(static_obj.shape[0]):
            value = static_obj[x].cpu().numpy()
            if np.mean(value) == 0:
                continue
            STATIC_OBJ_list.append(value)
 
        for x in range(curb_obj.shape[0]):
            value = curb_obj[x].cpu().numpy()
            if np.mean(value) == 0:
                continue
            STATIC_OBJ_list.append(value)
        
        save_img_path = os.path.join(images_folder, str(scene_name) + '_' + str(ts_token) + '.jpg')
        input_trajectory = 'No match found.'
        gt_trajectory = torch.cat([torch.zeros(1, 1, 2), inputs[0]['pts_ego'][:,:,:2].cpu()], dim=1).squeeze(0).numpy()
        pred_trajectory = torch.cat([torch.zeros(1, 2), traj_pred_checker.cpu()], dim=0).numpy()
        bdm_goal = inputs[0]['bdm_goal']
        vis.vis_trajectory_emu3fast(input_trajectory, gt_trajectory, pred_trajectory, imgs_paths, save_img_path, OBJ_List, OBJ_List_Pred, STATIC_OBJ_list, bdm_goal)

# 主处理函数
def process_json_file(json_file_path, pkl_folder, images_folder):
    # 提取 sceneID 和 timestamp
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    scene_id = base_name.split('_')[0]
    timestamp = base_name.split('_')[1][:-2]

    # 读取 JSON 文件
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    action_pred = np.array([point for point in data['action']])
    action_gt = np.array([point for point in data['action_gt_denorm']])

    traj_pred = make_traj_from_segs(action_pred,0,1,2)
    traj_gt = make_traj_from_segs(action_gt,0,1,2)

    traj_pred = torch.from_numpy(traj_pred)
    traj_gt = torch.from_numpy(traj_gt)

    # 查找对应的 pkl 文件
    pkl_file = find_pkl_file(pkl_folder, scene_id, timestamp)
    if not pkl_file:
        return None

    # 读取 pkl 数据
    with open(pkl_file, "rb") as f:
        inputs = pickle.load(f)

    traj_pred_checker = interpolate_traj_3d(traj_pred, np.arange(0.5, 5.01, 0.5), np.arange(0.2, 5.01, 0.2))
    traj_gt_checker = interpolate_traj_3d(traj_gt, np.arange(0.5, 5.01, 0.5), np.arange(0.2, 5.01, 0.2))

    # 执行计算
    checkerMetric = EgoHmfCheckerPredict()
    result = checkerMetric.checkerPredict(traj_pred_checker, inputs)
    
    # if not result['Pred_Static_Collision_Rate_3s'].item():
        # print(str(result['save_dict']['scene_name'][0]) + str(result['save_dict']['ts_token'][0]))
    vvis_trajectory(traj_pred_checker, inputs, images_folder)
    return result

# 主函数
def main(args):
    json_folder = args.json_path  # 替换为你的 JSON 文件夹路径
    pkl_folder = args.pkl_path  # 替换为你的 PKL 文件夹路径
    image_folder = json_folder + '/evaluate_images/'
    shutil.rmtree(image_folder, ignore_errors=True) or os.makedirs(image_folder)

    results = []

    total_files = 0
    read_files = 0
    # 获取所有json文件
    json_files = [f for f in os.listdir(json_folder)][:100]
    for file in tqdm(json_files, desc="Processing files"):
        if file.endswith(".json"):
            json_path = os.path.join(json_folder, file)
            result = process_json_file(json_path, pkl_folder, image_folder)
            total_files += 1
            if result is not None:
                results.append(result)
                read_files += 1
    
    print('total files:' + str(total_files))
    print('read files:' + str(read_files))
            

    # 输出所有结果（示例）
    final_checker_dict = compute_averages(results, filter_key="is_valid_for_checker")
    print("All results:")
    print(final_checker_dict)  
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process trajectory JSON files.')
    parser.add_argument('--json_path', type=str, help='Path to folder containing JSON files')
    parser.add_argument('--pkl_path', type=str, help='Path to the large jsonl file')
    args = parser.parse_args()
    
    main(args)
