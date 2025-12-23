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
