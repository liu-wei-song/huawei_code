import cv2
import numpy as np
# from uvp_module.datasets.utils import INVALID_VELOCITY
# from uvp_module.datasets.utils import Obj, ObjGt

class DynamicObject:
    def __init__(self, res, has_velocity=False, has_movement=False, has_track=False, is_gt=False):
        """ Naive dynamic object class for visualization purposes
            Args:
                res (np.array): (9, ),
                    bbox3d results representation in the format
                    (x, y, z, l, w, h, heading, class, state)
                    if has_velocity, the size becomes (11, )
                    (x, y, z, l, w, h, heading, class, state, vx, vy)
                has_velocity (bool): whether res has velocity info
        """
        obj = Obj
        if is_gt:
            obj = ObjGt

        self.x = res[obj.x.value]
        self.y = res[obj.y.value]
        self.z = res[obj.z.value]
        self.lx = res[obj.lx.value]
        self.ly = res[obj.ly.value]
        self.lz = res[obj.lz.value]
        self.orien = res[obj.ry.value]
        self.label = res[obj.label.value]
        self.has_velocity = has_velocity
        if has_velocity:
            self.vx = res[Obj.vx.value]
            self.vy = res[Obj.vy.value]
        if has_movement:
            self.mov = res[obj.mov.value]
        if has_track:
            self.id = int(res[obj.id.value])

    def __repr__(self):
        msg = f"DynamicObject: center=[{self.x},{self.y},{self.z}], " + \
              f"size=[{self.lx},{self.ly},{self.lz}], " + \
              f"orientation={self.orien}, label={self.label}"
        if self.has_velocity:
            msg += f"vx={self.vx}, vy={self.vy}"
        return msg

    @property
    def vel(self):
        return (self.vx ** 2 + self.vy ** 2) ** 0.5


def prepare_bev_canvas(canvas_size, center_pix, meter_to_pix, **kwargs):
    w, h = canvas_size
    if 'static_pic' in kwargs:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        static_pic = kwargs.get('static_pic')
        w_, h_, _ = static_pic.shape
        canvas[:w_, :h_] = static_pic
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # draw center
    cv2.circle(canvas, tuple(center_pix), 5, (0, 0, 255), 3)

    # draw grid
    iters_x = int(canvas.shape[0] / (10 * meter_to_pix))
    iters_y = int(canvas.shape[1] / (10 * meter_to_pix))
    for i in range(0, iters_y + 1):
        dx1 = center_pix[0] - int(i * 10 * meter_to_pix)
        dx2 = center_pix[0] + int(i * 10 * meter_to_pix)
        cv2.line(canvas, (dx1, 0), (dx1, canvas.shape[0]), (255, 255, 255))
        cv2.line(canvas, (dx2, 0), (dx2, canvas.shape[0]), (255, 255, 255))
    for i in range(0, iters_x + 1):
        dy1 = center_pix[1] - int(i * 10 * meter_to_pix)
        dy2 = center_pix[1] + int(i * 10 * meter_to_pix)
        cv2.line(canvas, (0, dy1), (canvas.shape[1], dy1), (255, 255, 255))
        cv2.line(canvas, (0, dy2), (canvas.shape[1], dy2), (255, 255, 255))

    if kwargs['far_distance']:
        dx1 = center_pix[0] - int(44.8 * meter_to_pix)
        dx2 = center_pix[0] + int(44.8 * meter_to_pix)
        dy1 = center_pix[1] + int(110 * meter_to_pix)
        dy2 = center_pix[1] - int(150 * meter_to_pix)
 
        cv2.line(canvas, (dx1, dy1), (dx1, dy2), (86, 101, 115), 10)
        cv2.line(canvas, (dx2, dy1), (dx2, dy2), (86, 101, 115), 10)


    for i in range(-130, 310, 10):
        cv2.putText(canvas, str(i) + 'm', (center_pix[0] + int(45 * meter_to_pix), center_pix[1] - int(i * meter_to_pix)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 150), 2, cv2.LINE_AA)

    return canvas


def draw_dynamic_obj(obj, center_pix, meter_pix_scale, color,
                     canvas=None, border=2, velocity=None,
                     text=None, text_pos=None, text_color=None, ignored=False, **kwargs):

    x, y, lx, ly = obj.x, obj.y, obj.lx, obj.ly
    orien = -obj.orien
    pts = np.array([[lx / 2, ly / 2], [lx / 2, -ly / 2],
                    [-lx / 2, -ly / 2], [-lx / 2, ly / 2]])
    Ro = np.array([[np.cos(orien), np.sin(orien)],
                   [-np.sin(orien), np.cos(orien)]])
    pts = pts @ Ro.T + np.array([x, y])
    uvs = []
    for i in range(0, 4):
        u = int(center_pix[0] - pts[i][1] * meter_pix_scale)
        v = int(center_pix[1] - pts[i][0] * meter_pix_scale)
        cv2.circle(canvas, (u, v), 1, color, 1)
        uvs.append((u, v))

    # draw box
    cv2.line(canvas, uvs[0], uvs[1], color, border, 8)
    cv2.line(canvas, uvs[1], uvs[2], color, border, 8)
    cv2.line(canvas, uvs[2], uvs[3], color, border, 8)
    cv2.line(canvas, uvs[0], uvs[3], color, border, 8)

    # draw velocity
    if velocity is not None and not ignored:
        v_uv1 = ((uvs[0][0] + uvs[2][0]) // 2,
                 (uvs[0][1] + uvs[2][1]) // 2)
        # center offset by velocity vector; 1 m/s corresp. to 10 pixels
        v_scale = 10 if kwargs.get('show_velocity') else 5
        if obj.label in (2, 3):
            # if is ped or cyc, 1 m/s to 40 pixels
            v_scale *= 4
        v_uv2 = (int(v_uv1[0] - obj.vy * v_scale),
                 int(v_uv1[1] - obj.vx * v_scale))
        cv2.arrowedLine(canvas, v_uv1, v_uv2, color, thickness=2, tipLength=0.2)

    if text is not None and not ignored:
        name_2_pos = {
            'center': ((uvs[0][0] + uvs[3][0]) // 2, (uvs[0][1] + uvs[3][1]) // 2),
            'top_left': (uvs[0][0] - 50, uvs[0][1]),
            'top_right': (uvs[1][0], uvs[1][1]),
        }
        if text_pos is None or text_pos not in name_2_pos:
            text_pos = 'center'
        text_color = text_color or (0, 255, 0)
        cv2.putText(canvas, text, name_2_pos[text_pos],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    return canvas


def draw_dynamic_obj_contour(contour, obj, center_pix, meter_pix_scale,
                     canvas=None, text=None, text_pos=None, color=(0, 0, 255), border=2):
    x, y, lx, ly = obj.x, obj.y, obj.lx, obj.ly
    pts = np.array([[lx / 2, ly / 2], [lx / 2, -ly / 2],
                    [-lx / 2, -ly / 2], [-lx / 2, ly / 2]])
    orien = 0
    Ro = np.array([[np.cos(orien), np.sin(orien)],
                   [-np.sin(orien), np.cos(orien)]])
    pts = pts @ Ro.T + np.array([x, y])

    x_max, y_max = canvas.shape[:2]

    def u_clamp(x, min_v, max_v):
        return int(min(max(x, min_v), max_v))

    # 首尾相连
    contour = np.concatenate((contour, contour[:2])) - 0.5

    p_num = contour.shape[-1] // 2
    for i in range(1, p_num):
        prev_i = i - 1
        cur_u = int(center_pix[0] - (y - ly * contour[2*i + 1]) * meter_pix_scale)
        cur_v = int(center_pix[1] - (x - lx * contour[2*i]) * meter_pix_scale)
        prev_u = int(center_pix[0] - (y - ly * contour[2*prev_i + 1]) * meter_pix_scale)
        prev_v = int(center_pix[1] - (x - lx * contour[2*prev_i]) * meter_pix_scale)
        cv2.line(canvas,(cur_u, cur_v), (prev_u, prev_v),
                 color, border)

    uvs = []
    for i in range(0, 4):
        u = int(center_pix[0] - pts[i][1] * meter_pix_scale)
        v = int(center_pix[1] - pts[i][0] * meter_pix_scale)
        uvs.append((u, v))

    if text is not None:
        name_2_pos = {
            'center': ((uvs[0][0] + uvs[3][0]) // 2, (uvs[0][1] + uvs[3][1]) // 2),
            'top_left': (uvs[0][0] - 50, uvs[0][1]),
            'top_right': (uvs[1][0], uvs[1][1]),
        }
        if text_pos is None or text_pos not in name_2_pos:
            text_pos = 'center'
        text_color = color or (0, 255, 0)
        cv2.putText(canvas, text, name_2_pos[text_pos],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    return canvas

def draw_static_obj_bev(static_obj, center_pix, meter_pix_scale, color, canva):
    for i in range(static_obj.shape[0] - 1):
        point1 = (int(center_pix[0] - static_obj[i][1] * meter_pix_scale), int(center_pix[1] - static_obj[i][0] * meter_pix_scale))
        point2 =  (int(center_pix[0] - static_obj[i+1][1] * meter_pix_scale), int(center_pix[1] - static_obj[i+1][0] * meter_pix_scale))
        cv2.line(canva, point1, point2, color, 2, 8)
    return canva

def draw_dynamic_obj_bev(
        obj, center_pix, meter_pix_scale, color, canvas=None,
        border=2, draw_cls=False, conf=None,
        show_yaw=False, show_velocity=False, show_movement=False):
    """ Draw dynamic object under bev canvas
        Args:
            obj (DynamicObject): DynamicObject class object storing object info
            center_pix (list<int>): center pixel location in canvas
            meter_pix_scale (int): resolution, number of pixels for 1 meter
            color (tuple<int>): specified box color
            canvas (np.array): if canvas is not None, draw boxes on existing canvas
    """
    x = obj.x
    y = obj.y
    lx = obj.lx
    ly = obj.ly
    orien = -obj.orien
    pts = np.array([[lx / 2, ly / 2],
                    [lx / 2, -ly / 2],
                    [-lx / 2, -ly / 2],
                    [-lx / 2, ly / 2]])

    Ro = np.array([[np.cos(orien), np.sin(orien)],
                   [-np.sin(orien), np.cos(orien)]])
    pts = pts @ Ro.T + np.array([x, y])
    uvs = []
    for i in range(0, 4):
        u = int(center_pix[0] - pts[i][1] * meter_pix_scale)
        v = int(center_pix[1] - pts[i][0] * meter_pix_scale)
        cv2.circle(canvas, (u, v), 1, color, 1)
        uvs.append((u, v))
    # draw box
    cv2.line(canvas, uvs[0], uvs[1], color, border, 8)
    cv2.line(canvas, uvs[1], uvs[2], color, border, 8)
    cv2.line(canvas, uvs[2], uvs[3], color, border, 8)
    if False:
        cv2.arrowedLine(canvas, uvs[0], uvs[3], color, border, 8)
    else:
        cv2.line(canvas, uvs[0], uvs[3], color, border, 8)
    # draw object class
    if draw_cls is True:
        cv2.putText(canvas, f'C{int(obj.label)}', uvs[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # draw cls confidence
    if conf is not None:
        _u = (uvs[0][0] + uvs[3][0]) // 2
        _v = (uvs[0][1] + uvs[3][1]) // 2
        cv2.putText(canvas, f'{conf:.2f}', (_u, _v),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # draw velocity vector if any
    INVALID_VELOCITY = 50.0
    if obj.has_velocity and obj.vx < INVALID_VELOCITY:
        # object center
        v_uv1 = ((uvs[0][0] + uvs[2][0]) // 2,
                 (uvs[0][1] + uvs[2][1]) // 2)
        # center offset by velocity vector; 1 m/s corresp. to 5 pixels
        v_scale = 10 if show_velocity else 5
        if obj.label in (2, 3):
            v_scale *= 4
        v_uv2 = (int(v_uv1[0] - obj.vy * v_scale),
                 int(v_uv1[1] - obj.vx * v_scale))
        cv2.arrowedLine(canvas, v_uv1, v_uv2, color, thickness=2, tipLength=0.2)

    if show_yaw:
        is_gt = color[2] == 255
        u_ = uvs[0][0] - 50 if is_gt else uvs[1][0]
        v_ = uvs[0][1] if is_gt else uvs[1][1]
        heading = obj.orien
        if heading > np.pi:
            heading = heading % (np.pi)
        elif heading < -np.pi:
            heading = heading % (-np.pi)
        heading = heading + (heading < -np.pi / 2) * np.pi - (heading > np.pi / 2) * np.pi
        heading = f'{(heading * 180 / np.pi):.1f}'
        cv2.putText(canvas, heading, (u_, v_),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    if show_movement:
        is_gt = (color[2] == 255)
        u_ = uvs[0][0] - 50 if is_gt else uvs[1][0]
        v_ = uvs[0][1] if is_gt else uvs[1][1]
        movment = f'{int(obj.mov)}' if is_gt else f'{obj.mov:.1f}'
        cv2.putText(canvas, str(movment), (u_, v_),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return canvas


def draw_bbox3d_proj(img, bbox3d_proj_corners, color, thickness=2):
    box_8p = np.array(bbox3d_proj_corners).astype(np.int32)
    pts1 = box_8p[0]
    pts2 = box_8p[1]
    pts3 = box_8p[2]
    pts4 = box_8p[3]
    pts5 = box_8p[4]
    pts6 = box_8p[5]
    pts7 = box_8p[6]
    pts8 = box_8p[7]
    cv2.line(img, (pts1[0], pts1[1]), (pts2[0], pts2[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts2[0], pts2[1]), (pts3[0], pts3[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts3[0], pts3[1]), (pts4[0], pts4[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts4[0], pts4[1]), (pts1[0], pts1[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts5[0], pts5[1]), (pts6[0], pts6[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts6[0], pts6[1]), (pts7[0], pts7[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts7[0], pts7[1]), (pts8[0], pts8[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts8[0], pts8[1]), (pts5[0], pts5[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts1[0], pts1[1]), (pts5[0], pts5[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts2[0], pts2[1]), (pts6[0], pts6[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts3[0], pts3[1]), (pts7[0], pts7[1]), (0, 0, 255), thickness)
    cv2.line(img, (pts4[0], pts4[1]), (pts8[0], pts8[1]), (0, 0, 255), thickness)
    return img


def label_color_mapping(numpy_label):
    assert len(numpy_label.shape) == 2
    numpy_label = numpy_label.astype(np.uint8)
    cmap = np.zeros([23, 3], 'uint8')
    cmap[0, :] = np.array([0, 0, 0])
    cmap[1, :] = np.array([0, 0, 255])
    cmap[2, :] = np.array([0, 255, 0])
    cmap[3, :] = np.array([255, 0, 0])
    cmap[4, :] = np.array([128, 128, 128])
    cmap[5, :] = np.array([255, 0, 255])
    cmap[6, :] = np.array([0, 255, 255])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])
    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])
    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])
    ims = cmap[numpy_label, :].reshape([numpy_label.shape[0], numpy_label.shape[1], -1])
    return ims


def using_moxing(path):
    if isinstance(path, list):
        path = path[0]
    return path.startswith('obs://')


def draw_box(x, y, lx, ly, canvas, color, theta, vis_args):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    pts = np.array([[lx / 2 * cos_theta - ly / 2 * sin_theta, 
                        lx / 2 * sin_theta + ly / 2 * cos_theta], 
                    [lx / 2 * cos_theta + ly / 2 * sin_theta, 
                        lx / 2 * sin_theta - ly / 2 * cos_theta],
                    [- lx / 2 * cos_theta + ly / 2 * sin_theta, 
                        - lx / 2 * sin_theta - ly / 2 * cos_theta], 
                    [- lx / 2 * cos_theta - ly / 2 * sin_theta, 
                        - lx / 2 * sin_theta + ly / 2 * cos_theta]])
    pts = pts + np.array([x, y])
    uvs = []
    center_pix = vis_args['center_pix']
    meter_pix_scale = vis_args['meter_to_pix']
    for i in range(0, 4):
        u = int(center_pix[0] - pts[i][1] * meter_pix_scale)
        v = int(center_pix[1] - pts[i][0] * meter_pix_scale)
        uvs.append((u, v))

    # draw box
    border = 2
    cv2.line(canvas, uvs[0], uvs[1], color, border, 8)
    cv2.line(canvas, uvs[1], uvs[2], color, border, 8)
    cv2.line(canvas, uvs[2], uvs[3], color, border, 8)
    cv2.line(canvas, uvs[0], uvs[3], color, border, 8)
