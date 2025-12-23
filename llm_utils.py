import numpy as np
# from uvp_module.datasets.sdp_llm.cot.prompt_generator import PromptGenerator, ResponseType
from autoscenes_plotter import CameraCanvasCombiner
import matplotlib.pyplot as plt
import moxing as mox
from vis_utils import prepare_bev_canvas, draw_dynamic_obj_bev, draw_static_obj_bev
import cv2
import os
import glob
import argparse
# from valley.utils import get_rank
import torch
from safetensors.torch import load_file
# from lib.utils.state_dict_utils import torch_load_moxing
# from uvp_module.datasets.navi_nn.utils_pln.onemap_data import OneMapMainAction, OneMapAssistantAction, OneMapLaneAction
from PIL import Image, ImageFont, ImageDraw
import io

import re
BUCKET_AREA = os.environ.get('BUCKET_AREA', None)
def remote_path(x, use_guiyang_bucket2=False):
    if not x.startswith('obs://'):
        # 如果不是s3的路径(obs://开通)，就说明是local path
        return x

    if 'model-artifact' in x:
        return x

    bucket_area = BUCKET_AREA

    if bucket_area == "beijing":
        return re.sub('obs://.*?/', 'obs://ads-training-beijing/', x)
    elif bucket_area == 'wulan':
        if 'ads-ascend-battle-y' in x:
            return x
        if 'ads-cloud-gy-y' in x:
            return x.replace('ads-cloud-gy-y', 'ads-ascend-battle-y')
        return re.sub('obs://.*?/', 'obs://ads-ascend-battle-y/', x)
    elif bucket_area == 'god':
        return x.replace('obs://god-training-data', 'obs://alluxio-131')
    elif bucket_area == 'suzhou':
        return re.sub('obs://.*?/', 'obs://god-training-data-sz/', x)
    elif bucket_area == "guiyang":
        if 'ads-cloud-gy-y' in x or 'prediction-analysis-data-gy' in x:
            return x
        if 'ads-ascend-battle-y' in x:
            if use_guiyang_bucket2 or 'data/prediction/feature' in x:
                return re.sub('obs://.*?/', 'obs://prediction-analysis-data-gy/', x)
            return x.replace('ads-ascend-battle-y', 'ads-cloud-gy-y')
        x = x.replace("obs://god-training-data/data/god/autoscenes/", "obs://god-training-data/data/god/autoscenes-prod/")
        return re.sub('obs://.*?/', 'obs://ads-cloud-gy-y/', x)
    elif bucket_area == 'shanghai':
        return re.sub('obs://.*?/', 'obs://alluxio-131/', x)
    elif bucket_area == 'hnt2out-guiyang':
        if 'yw-ads-training-2-gy1' in x:
            return x
        return re.sub('obs://.*?/','obs://yw-ads-training-gy1/', x)
    else:
        return x

import pyarrow as pa
import pyarrow.parquet as pq
def parquet(file_path, default=None):
    file_path = remote_path(file_path)
    with io.BytesIO(mox.file.read(file_path, binary=True)) as f:
        file_data = f.read()
    table = pq.read_table(pa.py_buffer(file_data)).to_pandas()
    return table

class DynamicObject:
    def __init__(self, value, has_velocity=False, has_movement=False):
        self.x = value[0]
        self.y = value[1]
        self.z = value[2]
        self.lx = value[3]
        self.ly = value[4]
        self.lz = value[5]
        self.orien = value[6]
        self.label = value[7]
        self.vx = value[9]
        self.vy = value[10]
        self.has_velocity = has_velocity


class CaseVisualization(object):
    def __init__(self):
        plt.rcParams['figure.figsize'] = (40.96, 20.48)
        self.surrounding_camera_combiner = CameraCanvasCombiner()
        self.img_len = 6
        self.vis_args = dict(center_pix=[720, 1550],
                        canvas_size=[1440, 2400],
                        meter_to_pix=17,
                        far_distance=True)

    def parse_model_output(self, predict_text, label_text, input_text, task_i='T1'):
        res_type = PromptGenerator.prompt_response_type[task_i]
        gt_success, gt_trajectory = PromptGenerator.parse_output(label_text, res_type)
        pred_success, pred_trajectory = PromptGenerator.parse_output(predict_text, res_type)
        _, input_trajectory = PromptGenerator.parse_output(input_text, res_type)
        success = True
        if not pred_success:
            print('The format of prediction is incorrect and cannot be parsed!!!')
            success = False
        if not gt_success:
            print('The format of ground truth is incorrect and cannot be parsed!!!')
            success = False
        if success:
            gt_trajectory = np.concatenate((np.array([[0.0, 0.0]]), gt_trajectory), axis=0)
            pred_trajectory = np.concatenate((np.array([[0.0, 0.0]]), pred_trajectory), axis=0)
        return success, input_trajectory, gt_trajectory, pred_trajectory

    # visualize trajectory from text output
    def vis_trajectory_emu3fast(self, input_trajectory, gt_trajectory, pred_trajectory, imgs_paths, save_img_path, OBJ_List, OBJ_List_Pred, STATIC_OBJ_list, bdm_goal=None,
                        total_additional_values_list=None, task_i='T1', P_veh2img_dict=None,
                        input_text_vis=None, out_text_vis=None, total_sd_feature_list=None, total_tbt_current_list=None,
                        total_tbt_prev_list=None, total_tbt_future_list=None, total_tbt_current_lane=None, vis_navigation=False,
                        cot_txt=None, cot_label=None, cot_output=None):
        text_dict = {}

        canvas = self.vis_trajectory_vector(input_trajectory, gt_trajectory, pred_trajectory, imgs_paths, OBJ_List,OBJ_List_Pred,STATIC_OBJ_list, bdm_goal, P_veh2img_dict, text_dict,
                                                total_additional_values_list, total_sd_feature_list, total_tbt_current_list, total_tbt_prev_list,
                                                total_tbt_future_list, total_tbt_current_lane, vis_navigation,cot_txt, cot_label, cot_output)
        self.save_image(canvas, save_img_path)

    # visualize trajectory from text output
    def vis_trajectory(self, predict, label, input, imgs_paths, save_img_path, OBJ_List,OBJ_List_Pred,STATIC_OBJ_list, bdm_goal=None,
                        total_additional_values_list=None,task_i='T1', P_veh2img_dict=None,
                        input_text_vis=None, out_text_vis=None, total_sd_feature_list=None, total_tbt_current_list=None,
                        total_tbt_prev_list=None, total_tbt_future_list=None, total_tbt_current_lane=None, vis_navigation=False,
                        cot_txt=None, cot_label=None, cot_output=None):
        text_dict = {}
        if input_text_vis is not None:
            text_dict['input_text'] = input_text_vis
        if out_text_vis is not None:
            text_dict['output_text'] = out_text_vis

        if isinstance(input, str):
            success, input_trajectory, gt_trajectory, pred_trajectory = self.parse_model_output(predict, label, input, task_i)
            if not success:
                return
            canvas = self.vis_trajectory_vector(input_trajectory, gt_trajectory, pred_trajectory, imgs_paths, OBJ_List,OBJ_List_Pred,STATIC_OBJ_list, bdm_goal, P_veh2img_dict, text_dict,
                                                    total_additional_values_list, total_sd_feature_list[0], total_tbt_current_list, total_tbt_prev_list,
                                                    total_tbt_future_list, total_tbt_current_lane, vis_navigation,cot_txt, cot_label, cot_output)
            self.save_image(canvas, save_img_path)
        else:
            canvas = self.vis_trajectory_vector(input, label, np.array([]), imgs_paths, OBJ_List,OBJ_List_Pred,STATIC_OBJ_list, P_veh2img_dict, text_dict,
                                                    total_additional_values_list, total_sd_feature_list[0], total_tbt_current_list, total_tbt_prev_list,
                                                    total_tbt_future_list, total_tbt_current_lane, vis_navigation,cot_txt, cot_label, cot_output)
            self.save_image(canvas, save_img_path)

    # visualize trajectory from vectorized trajectory
    def vis_trajectory_vector(self, input_trajectory:np.ndarray, gt_trajectory :np.ndarray, pred_trajectory:np.ndarray,
                              imgs_paths, OBJ_List,OBJ_List_Pred,STATIC_OBJ_list, bdm_goal=None, P_veh2img_dict=None, text_dict=None,
                              additional_values_list=None, sd_feature=None, tbt_current=None, tbt_prev=None,
                              tbt_future=None, tbt_current_lane=None, vis_navigation=False,cot_txt=None, cot_label=None, cot_output=None):
        plt.figure()
        cam_dict = {}
        for img_index in range(6):
            if img_index < self.img_len:
                if imgs_paths[img_index].endswith('parquet'):
                    img_index2id = {0:1, 1:2, 2:4, 3:5, 4:6, 5:7}
                    img_pq = parquet(imgs_paths[img_index])
                    cam_column = f"cam_{img_index2id[img_index]}" 
                    data = img_pq[cam_column].values[0]
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                    cam_dict[f"{cam_column}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_np = np.frombuffer(mox.file.read(imgs_paths[img_index], binary=True), np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
                    cam_dict["cam_" + imgs_paths[img_index].split('cam_')[-1][0]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # projection of trajectory to cam_1
        if P_veh2img_dict is not None:
            cam1_gt_traj, _ = cam_1_projector(cam_dict, gt_trajectory, P_veh2img_dict, clip_invalid=True)
            draw_trajectory((-cam1_gt_traj[:, [1, 0]]).tolist(), [0, 0], 1, (255, 0, 0), cam_dict['cam_1'], 'circle')
            cam1_pred_traj, _ = cam_1_projector(cam_dict, pred_trajectory, P_veh2img_dict, clip_invalid=True)
            draw_trajectory((-cam1_pred_traj[:, [1, 0]]).tolist(), [0, 0], 1, (0, 255, 0), cam_dict['cam_1'], 'cross')
        combined_img = self.surrounding_camera_combiner.surrounding_camera_combine(cam_dict)
        if 'input_text' in text_dict:
            self.visualize_model_text(combined_img, text_dict['input_text'], self.surrounding_camera_combiner.side_front_blank_box)
        if 'output_text' in text_dict:
            self.visualize_model_text(combined_img, text_dict['output_text'], self.surrounding_camera_combiner.side_back_blank_box)
        plt.subplot(1, 2, 1)
        plt.imshow(combined_img, interpolation='nearest', aspect='auto')
        plt.subplot(1, 2, 2)
        dynamic_color = (255, 255, 0)
        dynamic_color_pred = (0, 255, 255)
        # draw bev trajectory and objects
        canvas = prepare_bev_canvas(**self.vis_args)
        for obj in OBJ_List:
            canvas = draw_dynamic_obj_bev(obj, self.vis_args['center_pix'], self.vis_args['meter_to_pix'], dynamic_color, canvas,show_velocity=True)
        for obj in OBJ_List_Pred:
            canvas = draw_dynamic_obj_bev(obj, self.vis_args['center_pix'], self.vis_args['meter_to_pix'], dynamic_color_pred, canvas,show_velocity=True)

        static_color = (128, 128, 128)
        for static_obj in STATIC_OBJ_list:
            canvas = draw_static_obj_bev(static_obj, self.vis_args['center_pix'], self.vis_args['meter_to_pix'], static_color, canvas)
        # draw_trajectory(input_trajectory.tolist(), self.vis_args['center_pix'], self.vis_args['meter_to_pix'], (125, 125, 125),
        #                 canvas, 'circle')

        draw_trajectory(gt_trajectory.tolist(), self.vis_args['center_pix'], self.vis_args['meter_to_pix'], (255, 0, 0), canvas, 'circle')
        show_legend(canvas, (50, 50), (255, 0, 0), legend_text='--gt')

        draw_trajectory(pred_trajectory.tolist(), self.vis_args['center_pix'], self.vis_args['meter_to_pix'], (0, 255, 0), canvas, 'cross')
        show_legend(canvas, (50, 90), (0, 255, 0), legend_text='--pred')

        if bdm_goal is not None:
            draw_trajectory(bdm_goal.reshape(-1,2).tolist(), self.vis_args['center_pix'], self.vis_args['meter_to_pix'], (255, 102, 102), canvas, 'circle')
            show_legend(canvas, (50, 130), (255, 102, 102), legend_text='--bdm')

        if additional_values_list is not None:
            input_text="additional values: "
            for value in additional_values_list:
                input_text+="{:.1f}, ".format(value)
            show_legend(canvas, (50, 170), (255, 0, 0), legend_text=input_text)
            # self.visualize_model_text(combined_img,input_text, self.surrounding_camera_combiner.side_front_blank_box)

        if vis_navigation:
            draw_trajectory(sd_feature.tolist(), self.vis_args['center_pix'], self.vis_args['meter_to_pix'], (255, 255, 0), canvas, 'circle')
            show_legend(canvas, (50, 210), (255, 255, 0), legend_text='--sd_road')  # 黄色
 
            if tbt_prev[0][0] >= 0:
                prev_text = 'prev: ' + ' '.join(
                    OneMapMainAction(tbt_prev[0][0]).name.split('ACTION_')[1].lower().split('_'))
            else:
                prev_text = f"prev_value: {tbt_prev[0][0]}"
            if tbt_current[0][0] > 0:
                current_text = 'current: ' + ' '.join(
                    OneMapMainAction(tbt_current[0][0]).name.split('ACTION_')[1].lower().split(
                        '_')) + f" in the next {tbt_current[0][2]} meters."
            else:
                current_text = f"current_value: {tbt_current[0][0]}"
            if tbt_current_lane.shape[1] > 0:
                tbt_lane_text = f"Please take the {tbt_current_lane[0] + 1} lane on the left."
                current_text += tbt_lane_text
            if tbt_future[0][0] >= 0:
                future_text = 'future: ' + ' '.join(
                    OneMapMainAction(tbt_future[0][0]).name.split('ACTION_')[1].lower().split('_'))
            else:
                future_text = f"future_value: {tbt_future[0][0]}"
 
            cv2.putText(canvas, prev_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, current_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, future_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
        if cot_txt is not None:
            row_start=450 if vis_navigation else 210
            row_step=50
            string_step=80
            while len(cot_txt)>0:
                cv2.putText(canvas, cot_txt[:min(string_step,len(cot_txt))], (50, row_start), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                row_start=row_start+row_step
                cot_txt=cot_txt[min(string_step,len(cot_txt)):]

        if cot_output is not None and cot_label is not None:
            tmp = Image.fromarray(canvas)
            fontStyle = ImageFont.truetype("./simsun.ttc", 30, encoding="utf-8")
            draw = ImageDraw.Draw(tmp)
            draw.text((30, 250), '标签:', (0, 255, 0), font=fontStyle)
            crop_len = 70
            i = 0
            while(len(cot_label)>0):
                tmp_cot = cot_label[:crop_len] if len(cot_label) > crop_len else cot_label
                draw.text((90, 250+i*35), tmp_cot, (0, 255, 0), font=fontStyle)
                i += 1
                cot_label = cot_label[crop_len:] if len(cot_label) > crop_len else ''

            draw.text((30, 250+i*35), '预测:', (0, 255, 255), font=fontStyle)
            while(len(cot_output)>0):
                tmp_cot = cot_output[:crop_len] if len(cot_output) > crop_len else cot_output
                draw.text((90, 250+i*35), tmp_cot, (0, 255, 255), font=fontStyle)
                i += 1
                cot_output = cot_output[crop_len:] if len(cot_output) > crop_len else ''
            canvas = np.asarray(tmp)
        return canvas

    def visualize_model_text(self, canvas, input_text, bbox):
        x0, x1, y0, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        import textwrap
        lines = textwrap.wrap(input_text, width=20)
        for i, text_line in enumerate(lines):
            canvas[y0:y1, x0:x1] = cv2.putText(canvas[y0:y1, x0:x1], text_line, (0, 25+25*i), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        return True

    def save_image(self, canvas, save_img_path):
        plt.imshow(canvas, interpolation='nearest', aspect='auto')
        plt.savefig(save_img_path, transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close('all')

def drarw_point(canvas,  uv, type='circle', color=(0, 0, 0)):
    #  draw cross symbol
    def draw_cross_mark(canvas, center_points, radius, color):
        start_point1 = (int(center_points[0] - radius/2.0), int(center_points[1] + radius/2.0))
        end_point1 = (int(center_points[0] + radius/2.0), int(center_points[1] - radius/2.0))
        start_point2 = (int(center_points[0] + radius/2.0), int(center_points[1] + radius / 2.0))
        end_point2 = (int(center_points[0] - radius/2.0), int(center_points[1] - radius / 2.0))
        cv2.line(canvas, start_point1, end_point1, color, 3, 8)
        cv2.line(canvas, start_point2, end_point2, color, 3, 8)

    if type == 'circle':
        if color == (255, 0, 0):
            cv2.circle(canvas,  uv, 8, color, -1)
        else:
            cv2.circle(canvas,  uv, 6, color, -1)
    else:
        draw_cross_mark(canvas,  uv, 12, color)
    return canvas

def draw_trajectory(trajectory, center_pix, meter_pix_scale, color, canvas, type):
    uvs = []

    # Draw points
    for i in trajectory:
        pts = np.array(i)
        u = int(center_pix[0] - pts[1] * meter_pix_scale)
        v = int(center_pix[1] - pts[0] * meter_pix_scale)
        drarw_point(canvas, (u, v), type, color)
        uvs.append((u, v))

    # Draw trajectory
    for point_index in range(len(uvs) - 1):
        cv2.line(canvas, uvs[point_index], uvs[point_index + 1], color, 2, 8)

    return canvas

def show_legend(canvas, uv, color, legend_text='--gt'):
    return cv2.putText(canvas, legend_text, uv, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


def images_to_video(path, clip_list=['e9e9990f06a64ee081bb81d797911bdd'],
                    save_dir='/home/ma-user/code/ssw/output'):
    for clip_name in clip_list:
        img_arrary = []
        for filename in sorted(glob.glob(os.path.join(path, f'{clip_name}*.jpg'))):
            img = cv2.imread(os.path.join(path, filename))
            img_arrary.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_dir, f'{clip_name}.mp4'), fourcc, 5, (4096, 2048))
        for i in img_arrary:
            i = cv2.resize(i, (4096, 2048))
            out.write(i)
        out.release()
   
def load_ckpt_old(model_path, model, strict=True, load_module=list(), load_teacher_module=list()):
    # load_module：主要用来控制load一阶段的模型的模块，如load_module=['egoHmf', 'modal_bridge']
    print('*************************************Loading Checkpoints*************************************')
    no_match_keys = []
    tmp_dict = dict()
    for i in os.listdir(model_path):
        if i.endswith('.bin') and i.startswith('pytorch_model'):
            state_dict = torch_load_moxing(os.path.join(model_path, i), map_location=torch.device('cuda'))
        elif i.endswith('.safetensors'):
            state_dict = load_file(os.path.join(model_path, i))
        else:
            continue
        if not load_module:
            for key in state_dict.keys():
                if key not in model.state_dict().keys() or state_dict[key].numel()!=model.state_dict()[key].numel():
                    no_match_keys.append(key)
            for key in no_match_keys:
                if key in state_dict.keys():
                    state_dict.pop(key)
            tmp_dict.update(state_dict)
        else:
            for key in state_dict.keys():
                module_name = key.split('.')[0]
                if module_name in load_module and state_dict[key].numel()==model.state_dict()[key].numel():
                    tmp_dict[key] = state_dict[key]              
            not_load_module = set()
            for key in model.state_dict().keys():
                if key not in tmp_dict.keys() and state_dict[key].numel()==model.state_dict()[key].numel():
                    tmp_dict[key] = model.state_dict()[key]
                    not_load_module.add(key.split('.')[0])
    if not load_module:
        print("the no match keys is: ", no_match_keys)
    else:
        if 'pts_bbox_head.sparse_head.loss_movement_class.weight' in tmp_dict.keys():
            tmp_dict.pop('pts_bbox_head.sparse_head.loss_movement_class.weight')
        print('------------------not load module------------------\n', not_load_module)

    model.load_state_dict(tmp_dict, strict=strict)
    print('*************************************End Checkpoints*************************************')
    return model

def keyTransfer_for_LAsplit(state_dict, model_state_dict, key):
    key_decouple=''
    if 'Model_Bridge_Head.'in key:
        key_decouple=key_decouple.replace('Model_Bridge_Head.','Model_Bridge_Head_Decouple.')
    if 'pdpn_decoder.'in key:
        key_decouple=key_decouple.replace('pdpn_decoder.','pdpn_decoder_decouple_v4_dense.')

    if key_decouple not in state_dict.keys() and key_decouple in model_state_dict.keys():
        state_dict[key_decouple]=state_dict[key]
    return state_dict
    
def keyTransfer_for_KVcache(state_dict, model_state_dict, key):
    if 'llm.base_model.model.model.layers.'in key or 'llm.model.layers.' in key:
        key_ActionExpert=key.replace('llm.base_model.model.model.layers.','fem_navi_head.ActionExpert.ActionExpert_block.')
        key_ActionExpert=key.replace('llm.model.layers.','fem_navi_head.ActionExpert.ActionExpert_block.')
        if '.self_attn.q_proj.weight'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.q_proj.weight','.query_transform.weight')
        if '.self_attn.q_proj.bias'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.q_proj.bias','.query_transform.bias')
        if '.self_attn.k_proj.weight'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.k_proj.weight','.key_transform.weight')
        if '.self_attn.k_proj.bias'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.k_proj.bias','.key_transform.bias')
        if '.self_attn.v_proj.weight'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.v_proj.weight','.value_transform.weight')
        if '.self_attn.v_proj.bias'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.v_proj.bias','.value_transform.bias')
        if '.self_attn.o_proj.weight'in key:
            key_ActionExpert=key_ActionExpert.replace('.self_attn.o_proj.weight','.final_transform.weight')

        if key_ActionExpert not in state_dict.keys() and key_ActionExpert in model_state_dict.keys():
            state_dict[key_ActionExpert]=state_dict[key]
    return state_dict

def load_ckpt(model_path, model, strict=True, load_module=None):
    """
    加载预训练参数, 适配多任务/一阶段/二阶段
    
    :param model_path: 加载预训练参数路径, str
    :param model: 加载模型对象
    :param strict: 严格匹配, bool
    :param load_module: 加载模块名成，默认None=全加载, list
    :type param: [type]
    :return: 更新参数后模型对象
    """
    print('*************************************My  load  model*************************************')
    no_match_keys_inckpt = []
    state_dict = dict()
    not_load_module_inckpt = set()
    not_load_module_inDriver = set()
    local_device = 'cpu' if os.getenv('USING_ASCEND_910B') == "1" else 'cuda'
    model_keys = list(model.state_dict().keys())
    model_state_dict = model.state_dict()
    tmp_dict = model.state_dict().copy()
    if not os.path.isdir(model_path):
        state_dict = torch.load(model_path, map_location=torch.device(local_device))['state_dict']
        tmp_dict.update(state_dict)
    else:
        for i in os.listdir(model_path):
            if i.endswith('.bin') and i.startswith('pytorch_model'):
                state_dict = torch.load(os.path.join(model_path, i), map_location=torch.device(local_device))
            elif i.endswith('.safetensors'):
                state_dict = load_file(os.path.join(model_path, i), device=local_device)
            else:
                continue        
            
            for key in list(state_dict.keys()):  # TODO:适配旧版模型加载, 择日删除
                if "decoder.mlp.layers." in key:
                    state_dict[key.replace('mlp', 'traj')] = state_dict[key]  
                # pdpn
                elif 'egoHmfBackbone'in key and 'pdpn_encoder' not in key:
                    state_dict[key.replace('egoHmfBackbone','pdpn_encoder.SdTransformerBackbone')]=state_dict[key]
                elif 'FSDdecoderBridge'in key and 'pdpn_decoder' not in key:
                    state_dict[key.replace('FSDdecoderBridge','pdpn_decoder.PDPNDecoderBridge.pdpn_decoder_bridge')]=state_dict[key]
                elif 'egoHmf'in key and 'pdpn_decoder.EgoHmfHead' not in key:
                    state_dict[key.replace('egoHmf','pdpn_decoder.EgoHmfHead')]=state_dict[key]
                elif 'egoHmfPredict'in key and 'pdpn_decoder.EgoHmfPredict' not in key:
                    state_dict[key.replace('egoHmfPredict','pdpn_decoder.EgoHmfPredict')]=state_dict[key]
                elif 'EgoHmfCheckerPredict'in key and 'pdpn_decoder.EgoHmfCheckerPredict' not in key:
                    state_dict[key.replace('EgoHmfCheckerPredict','pdpn_decoder.EgoHmfCheckerPredict')]=state_dict[key]
                # modal_bridge
                elif 'fsd_backbone_bridge_net'in key and 'model_bridge.pdpn_adapter' not in key:
                    state_dict[key.replace('fsd_backbone_bridge_net','model_bridge.pdpn_adapter')]=state_dict[key]
                elif 'MLP_adapter_BEV_into_llm'in key and 'model_bridge.MLP_BEV_adapter' not in key:
                    state_dict[key.replace('MLP_adapter_BEV_into_llm','model_bridge.MLP_BEV_adapter')]=state_dict[key]
                elif 'MLP_adapter_Vehicle_into_llm'in key and 'model_bridge.MLP_Vehicle_adapter' not in key:
                    state_dict[key.replace('MLP_adapter_Vehicle_into_llm','model_bridge.MLP_Vehicle_adapter')]=state_dict[key]
                elif 'MLP_adapter_staticRG_into_llm'in key and 'model_bridge.MLP_staticRG_adapter' not in key:
                    state_dict[key.replace('EgoHmfCheckerPredict','model_bridge.MLP_staticRG_adapter')]=state_dict[key]
                elif 'pdpn_simpleNet_bridge_bev'in key and 'model_bridge.MLP_BEV_adapter' not in key:
                    state_dict[key.replace('pdpn_simpleNet_bridge_bev','model_bridge.MLP_BEV_adapter')]=state_dict[key]
                elif 'pdpn_simpleNet_bridge_vehicle'in key and 'model_bridge.MLP_Vehicle_adapter' not in key:
                    state_dict[key.replace('pdpn_simpleNet_bridge_vehicle','model_bridge.MLP_Vehicle_adapter')]=state_dict[key]
                elif 'pdpn_simpleNet_bridge_staticRG'in key and 'model_bridge.MLP_staticRG_adapter' not in key:
                    state_dict[key.replace('pdpn_simpleNet_bridge_staticRG','model_bridge.MLP_staticRG_adapter')]=state_dict[key]
                
                # for LA分离 
                state_dict = keyTransfer_for_LAsplit(state_dict, tmp_dict, key)
                # for kv cache 
                state_dict = keyTransfer_for_KVcache(state_dict, tmp_dict, key)
        
            tmp_dict.update(state_dict)
            for key in list(state_dict.keys()):
                if key not in model.state_dict().keys() or state_dict[key].numel() != model_state_dict[key].numel() or (load_module and key.split('.')[0] not in load_module):
                    tmp_dict.pop(key)
                    no_match_keys_inckpt.append(key)
                    
            for key in model.state_dict().keys():
                if key not in tmp_dict.keys() or tmp_dict[key].numel()!=model_state_dict[key].numel():
                    not_load_module_inDriver.add(key)
    if 'pts_bbox_head.sparse_head.loss_movement_class.weight' in tmp_dict.keys():
        tmp_dict.pop('pts_bbox_head.sparse_head.loss_movement_class.weight')
    print('------------------no_match_keys in ckpt------------------\n', no_match_keys_inckpt)
    print('------------------not load module in ckpt------------------\n', not_load_module_inckpt)
    model.load_state_dict(tmp_dict, strict=strict)
    return model

def print_log(**kwargs):
    input_text = kwargs.get('input_text')
    label_trajectory = kwargs.get('label_text')
    output_trajectory = kwargs.get('output_text')
    ts_token = kwargs.get('ts_token')
    scene_name = kwargs.get('scene_ids')
    planning_pkl = kwargs.get('planing_pkl')
    if isinstance(label_trajectory, str):
        if get_rank() == 0:
            print('=' * 200)
            print('=== ts_token: ' + ts_token + ';' + 'sceneid: ' + scene_name)
            print('=== Input: ' + input_text)
            if planning_pkl:
                print('=== planing_pkl: ' + str(planning_pkl))
            if isinstance(output_trajectory, dict):
                for key, value in output_trajectory.items():
                    print(f"=== {key}: " + value.strip())
            else:
                print(f"=== Predict: " + output_trajectory)
            print("=== Label:  " + label_trajectory.strip())
    elif isinstance(label_trajectory, list):
        for bs in range(len(label_trajectory)):
            if get_rank() == 0:
                print('=' * 200)
                print('=== ts_token: ' + ts_token[bs] + ';' + 'sceneid: ' + scene_name[bs])
                print('=== Input: ' + input_text[bs])
                if planning_pkl:
                    print('=== planing_pkl: ' + str(planning_pkl[bs]))
                if isinstance(output_trajectory[bs], dict):
                    for key, value in output_trajectory[bs].items():
                        print(f"=== {key}: " + value.strip())
                else:
                    print(f"=== Predict: " + output_trajectory[bs])
                print("=== Label:  " + label_trajectory[bs].strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='/home/ma-user/code/tangchufeng/logs/llm4drive_train_a100_1node_0126_ClipQformer_v1_checkpoint-60000.log')
    args = parser.parse_args()
    return args


def tensor_traj_to_str(pred):
    pred_list = []
    for i in pred:
        if i[0].item() < -999:
            continue
        pred_list.append((round(i[0].item(),2), round(i[1].item(),2)))
    pred_str = str(pred_list)
    pred_str = pred_str.replace(' ','') # remove spaces
    return pred_str


if __name__ == '__main__':
    args = parse_args()
    with open(args.log_path, 'r') as f:
        for line in f:
            print(line)
