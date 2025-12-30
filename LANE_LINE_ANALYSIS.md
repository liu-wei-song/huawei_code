# 车道线处理逻辑完整分析

## 📊 数据流总览

```
PKL文件 → adsData.py → transfuser_gt_utils_ads.py → BEV语义图
   ↓
checkerPredict_vis.py / llm_utils.py → vis_utils.py → 可视化
```

---

## 1️⃣ 数据源格式（PKL文件）

```python
# 从 PKL 文件读取
cur_pkl_data = {
    'static_obj_feat': Tensor[N, P, 3],  # N条线，P个点，3维[x, y, ?]
    'static_obj_mask': Tensor[N, P],     # 点级mask
    'curb_feat': Tensor[K, P, 2],        # K条路缘
    'curb_mask': Tensor[K, P],
    'object_feat': ...,                   # 动态物体
}
```

**关键发现：**
- ✅ `static_obj_feat` 确实是 `(N, 20, 3)` 格式
- ✅ `static_obj_mask` 是 `(N, 20)` 点级mask
- ✅ 第三维含义不明，需要忽略

---

## 2️⃣ checkerPredict_vis.py 处理方式

```python
# hwcode/checkerPredict_vis.py:217-225
static_obj = inputs[1]['static_obj_feat'] * inputs[1]['static_obj_mask'][..., -1].unsqueeze(-1).unsqueeze(-1)
curb_obj = inputs[1]['curb_feat'] * inputs[1]['curb_mask'][..., -1].unsqueeze(-1).unsqueeze(-1)
static_obj, curb_obj = static_obj[key_frame_index], curb_obj[key_frame_index]

STATIC_OBJ_list = []
for x in range(static_obj.shape[0]):
    value = static_obj[x].cpu().numpy()  # shape: (P, 3)
    if np.mean(value) == 0:
        continue
    STATIC_OBJ_list.append(value)
```

**关键点：**
- ❌ 使用 `mask[..., -1]` - **只取每条线最后一个点的mask**
- 这导致如果最后一个点无效，整条线都被过滤
- **我们的新实现修复了这个问题**

---

## 3️⃣ llm_utils.py 绘制方式

```python
# huawei_code/llm_utils.py:193-198
static_color = (128, 128, 128)
for static_obj in STATIC_OBJ_list:
    canvas = draw_static_obj_bev(
        static_obj, 
        self.vis_args['center_pix'], 
        self.vis_args['meter_to_pix'], 
        static_color, 
        canvas
    )
```

**传递给 `draw_static_obj_bev` 的格式：**
- `static_obj`: shape `(P, 2)` 或 `(P, 3)` - 单条线的点序列

---

## 4️⃣ vis_utils.py 绘制实现

```python
# hwcode/vis_utils.py:193-199
def draw_static_obj_bev(static_obj, center_pix, meter_pix_scale, color, canva):
    for i in range(static_obj.shape[0] - 1):
        # 坐标转换：
        # x (前方) -> center_pix[1] - static_obj[i][0] * meter_pix_scale
        # y (左方) -> center_pix[0] - static_obj[i][1] * meter_pix_scale
        point1 = (int(center_pix[0] - static_obj[i][1] * meter_pix_scale), 
                  int(center_pix[1] - static_obj[i][0] * meter_pix_scale))
        point2 = (int(center_pix[0] - static_obj[i+1][1] * meter_pix_scale), 
                  int(center_pix[1] - static_obj[i+1][0] * meter_pix_scale))
        cv2.line(canva, point1, point2, color, 2, 8)
    return canva
```

**坐标系统：**
- Ego 在 `center_pix` 位置（底部中心）
- x 正向前 → 图像行向上（center_pix[1] - x）
- y 正向左 → 图像列向左（center_pix[0] - y）

---

## 5️⃣ transfuser_gt_utils_ads.py 新实现

```python
# hwcode/transfuser_gt_utils_ads.py
def _draw_polylines(
    bev_map, feat, mask, label, config, 
    thickness=2, scale=1, closed=False
):
    # ✅ 适配 (N, P, 3) 格式
    if feat.ndim == 3 and feat.shape[-1] == 3:
        feat = feat[..., :2]  # (N, P, 2)
    
    # ✅ 正确的 mask 处理
    if mask.ndim == 2:  # mask shape: (N, P)
        line_valid = mask.sum(axis=-1) >= 2  # 至少2个有效点
    
    for i in range(feat.shape[0]):
        if not line_valid[i]:
            continue
        
        points = feat[i]  # (P, 2)
        
        # ✅ 点级过滤
        if mask.ndim == 2:
            point_mask = mask[i].astype(bool)
            points = points[point_mask]
        
        # 过滤距离太小的点
        valid_mask = np.linalg.norm(points, axis=-1) > 0.01
        valid_points = points[valid_mask]
        
        if len(valid_points) < 2:
            continue
        
        # 转换为像素坐标
        pixel_coords = _world_to_pixel_array(valid_points, config, scale)
        
        # 绘制线条（带抗锯齿）
        cv2.polylines(bev_map, [pixel_coords], isClosed=closed,
                      color=int(label), thickness=max(1, thickness * scale),
                      lineType=cv2.LINE_AA)
```

---

## 6️⃣ adsData.py 集成方式

```python
# hwcode/adsData.py:1808-1823
# 2. 获取静态物体 (车道线等)
static_obj_feat = cur_pkl_data.get('static_obj_feat', None)  # (N, 20, 3)
static_obj_mask = cur_pkl_data.get('static_obj_mask', None)  # (N, 20)

if static_obj_feat is not None:
    static_obj_feat = np.array(static_obj_feat)
if static_obj_mask is not None:
    static_obj_mask = np.array(static_obj_mask)

# 3. 生成 BEV semantic map
bev_map = compute_bev_semantic_map_ads(
    obj_labels=obj_label,
    config=self.gt_config,
    static_obj_feat=static_obj_feat,  # ✅ 直接传递，函数内部处理
    static_obj_mask=static_obj_mask,
)
```

**优势：**
- ✅ 不需要预处理，直接传递原始数据
- ✅ 函数内部自动适配 `(N, P, 3)` 格式
- ✅ 正确处理 `(N, P)` mask

---

## 🔍 关键差异对比

| 项目 | 旧实现 (checkerPredict_vis) | 新实现 (transfuser_gt_utils_ads) |
|------|---------------------------|-----------------------------------|
| **mask 处理** | ❌ `mask[..., -1]` 只看最后一个点 | ✅ `mask.sum(axis=-1) >= 2` 判断有效点数 |
| **第三维处理** | ❌ 保留第三维（可能导致问题） | ✅ 自动忽略 `feat[..., :2]` |
| **点级过滤** | ❌ 无 | ✅ `point_mask = mask[i].astype(bool)` |
| **批量处理** | ❌ 单条线处理 | ✅ 批量处理N条线 |
| **坐标系** | Ego在底部中心 | ✅ Ego在图像中心（更合理） |
| **抗锯齿** | ❌ 无 | ✅ `cv2.LINE_AA` |

---

## ⚠️ 潜在问题

### 问题1：mask 处理不一致

**旧代码：**
```python
# ❌ 只检查最后一个点
mask[..., -1]  # shape: (N,)
```

**影响：**
- 如果一条线的第20个点是padding（mask=0），整条线被过滤
- 即使前19个点都有效也会被丢弃

**新代码已修复：**
```python
# ✅ 检查有效点总数
mask.sum(axis=-1) >= 2  # shape: (N,)
```

### 问题2：第三维含义不明

- `static_obj_feat[:, :, 2]` 的含义未知
- 可能是高度z，或其他特征
- **新代码已忽略此维度**

---

## ✅ 验证结果

```bash
Input shapes:
  static_obj_feat: (50, 20, 3) ✅
  static_obj_mask: (50, 20)    ✅

Output:
  BEV map shape: (256, 256)
  
  BEV Label Distribution:
    lane_lines: 1.25%  ✅ 成功绘制
    vehicles:   1.23%
    pedestrian: 0.05%
```

---

## 📝 建议

### 1. 统一 mask 处理逻辑

建议更新 `checkerPredict_vis.py` 使用更鲁棒的mask处理：

```python
# 改进建议
if static_obj_mask.ndim == 2:
    # 判断每条线是否有足够有效点
    line_valid = (static_obj_mask.sum(dim=-1) >= 2)
    static_obj_feat = static_obj_feat[line_valid]
```

### 2. 明确第三维用途

建议在数据文档中说明 `static_obj_feat[:, :, 2]` 的含义，或统一使用 `(N, P, 2)` 格式。

### 3. 坐标系统说明

当前有两种坐标系：
- **旧可视化**：Ego在底部中心（前方可视范围更大）
- **新BEV图**：Ego在图像中心（前后对称）

建议明确使用场景，避免混淆。

---

## 🎯 总结

新实现 `transfuser_gt_utils_ads.py` 相比旧的车道线处理逻辑：

✅ **优势**
1. 更鲁棒的 mask 处理
2. 自动适配多种数据格式
3. 批量处理效率更高
4. 更好的视觉效果（抗锯齿）
5. Ego中心坐标系更合理

⚠️ **需注意**
1. 坐标系不同（中心 vs 底部）
2. 可视化参数可配置（高分辨率、图例等）

🔧 **兼容性**
- 完全兼容原始 `(N, 20, 3)` 数据格式
- 完全兼容原始 `(N, 20)` mask 格式
- 可直接用于训练和可视化

