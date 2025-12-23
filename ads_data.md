bbox:
数据：object_feat(按照原始Emu3那一套)
可参考 obj_label[cnt] = torch.FloatTensor([
                center[0], center[1], center[2], size[0], size[1], size[2],
                heading, category, 0, vx, vy
            ])做成相应的数组

lane:
数据：cur_pkl_data['static_obj_feat']

object_feat
[obj_num, 51, 22]

障碍物信息[以下全部都是自车坐标系]
object_pose_ego.x, object_pose_ego.y, object_pose_ego.heading,
object_fusion_class, object_length, object_width, speed_limit,
object_vel, object_vel_orien, object_yaw_rate, object_is_static,
object_left_turn_light, object_right_turn_light,
object_box_pts[0]['x'], object_box_pts[0]['y'],
object_box_pts[1]['x'], object_box_pts[1]['y'],
object_box_pts[2]['x'], object_box_pts[2]['y'],
object_box_pts[3]['x'], object_box_pts[3]['y'],
object_is_ego

This image shows a section of code defining an enumeration class named `FusionClassification` in C++. The enumeration class is used to classify different types of objects detected by a system, likely for an autonomous vehicle's obstacle detection and classification system. Each classification is assigned a unique numeric value, and comments are provided to describe the type of object each value represents.

Here is the detailed breakdown of the enumeration:

- `CLASSIFICATION_UNKNOWN = 0U`: Represents an unknown classification.
- `CLASSIFICATION_MICRO_CAR = 1U`: Represents a small car (size level 1).
- `CLASSIFICATION_CAR = 2U`: Represents a regular car (size level 2).
- `CLASSIFICATION_VAN = 3U`: Represents a van (size level 3).
- `CLASSIFICATION_LIGHT_TRUCK = 4U`: Represents a light truck (size level 4).
- `CLASSIFICATION_TRUCK = 5U`: Represents a large truck (size level 5).
- `CLASSIFICATION_BUS = 6U`: Represents a large bus (size level 6).
- `CLASSIFICATION_PEDESTRIAN = 7U`: Represents a pedestrian (including children and those riding bicycles).
- `CLASSIFICATION_CYCLIST_BIKE = 8U`: Represents a cyclist riding a bicycle.
- `CLASSIFICATION_CYCLIST_MOTORCYCLE = 9U`: Represents a cyclist riding a motorcycle or electric bicycle.
- `CLASSIFICATION_MOTORCYCLE = 10U`: Represents a motorcycle (without a rider), reserved for future classification.
- `CLASSIFICATION_BICYCLE = 11U`: Represents a bicycle (without a rider), reserved for future classification.

The comments in Chinese provide additional context for each classification, indicating the type of vehicle or object and its size level. The term "预留暂不分" suggests that certain classifications are reserved for future use and are not currently classified.

cur_pkl_data:
dict_keys(['dis_to_cross', 'road_feature', 'road_class', 'anchors', 'object_position', 'right_boundary_mask', 'valid_route_num', 'right_boundary_feat', 'bdm_goal', 'left_boundary_mask', 'speed_limit', 'anchors_mask', 'pts_ego', 'left_boundary_feat', 'object_feat', 'cur_lane_idx', 'object_pred_mask', 'object_heading', 'gt_traj_3d', 'is_valid_for_checker', 'fem_path', 'fem_path_mask', 'curb_feat_', 'curb_mask_', 'good_follow_tag', 'follow_hist_dist', 'gt_traj', 'collision_index', 'steer_angle', 'follow_dist', 'pnc_pkl_valid_mask', 'ego_route', 'gt_target', 'thw_valid_data', 'label_center_back', 'is_back_obj', 'anchors_similarity_score', 'ego_gt_traj', 'traffic_graph', 'gt_target_cls_mask', 'gt_traj_mask', 'gt_target_class', 'gt_trj_in_ego', 'sample_tag', 'curr_intersection_feat', 'acc_data', 'hist_thw', 'exit_lane_label', 'ego_vel', 'ego_future_diverge', 'is_neg_sample', 'ego_heading', 'bdm_goal_fem', 'vru_pred_mask', 'origin_pos', 'is_center_back', 'gt_target_reg_mask', 'lane_center_dis', 'ego_lane_width', 'is_ego_in_edge_lane', 'dec_data', 'curvature_spd_limit', 'dis_to_road_end', 'veh_pred_mask', 'curb_dist', 'gt_traj_mask_ori', 'ego_cur_diverge', 'ego_v', 'lane_seq_label', 'anchor_lane_remain_dis', 'ego_s_mask', 'gt_traj_long', 'gt_traj_long_mask', 'is_gt_lane_change_on_reverse_laneRemain', 'is_merge_split', 'ego_in_converse_lane', 'cutin_obj_mask', 'cutin_obj_gm_mask', 'follow_obj_mask', 'ego_gt_mask', 'gt_traj_target', 'gt_traj_target_class', 'gt_traj_target_cls_mask', 'gt_traj_target_reg_class', 'gt_traj_candidate_target', 'gt_traj_target_mask', 'ped_pred_mask', 'full_gt_traj', 'full_gt_traj_trans', 'path_10m', 'path_10m_mask', 'neg_full_gt_traj_trans', 'neg_path_10m', 'neg_path_10m_mask', 'gt_diff', 'has_neg', 'is_lane_remain_valid', 'multigoal_top1', 'curb_feat', 'static_obj_mask', 'curb_mask', 'static_obj_feat'])

cur_pkl_data['object_feat'].shape
(64, 11, 21)
