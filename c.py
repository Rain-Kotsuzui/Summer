import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import open3d as o3d
from pynput import keyboard, mouse
import ctypes

# ==================== 1. 获取屏幕分辨率 ====================
user32 = ctypes.windll.user32
SCREEN_W, SCREEN_H = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
CENTER_X, CENTER_Y = SCREEN_W // 2, SCREEN_H // 2

# ==================== 新增：目标生命周期追踪器 (Tracker) ====================
class TrackedApple:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.hits = 1             # 连续命中次数
        self.misses = 0           # 连续丢失次数
        self.is_confirmed = False # 是否已确认为真实目标

    def update(self, center, radius):
        # EMA 平滑算法：让球体位置和大小更稳定，不抖动
        alpha = 0.6 
        self.center = alpha * center + (1 - alpha) * self.center
        self.radius = alpha * radius + (1 - alpha) * self.radius
        self.hits += 1
        self.misses = 0

    def predict_miss(self):
        # 没有匹配到时，增加丢失计数
        self.misses += 1

class AppleTracker:
    def __init__(self, dist_thresh=0.04):
        self.tracks = []
        self.dist_thresh = dist_thresh # 匹配的距离阈值 (10cm以内认为是同一个苹果)

    def update(self, detections, confirm_frames, max_lost_frames):
        unmatched_detections = list(detections)
        
        # 1. 尝试将新检测到的目标匹配到已有的跟踪器上
        for track in self.tracks:
            if not unmatched_detections:
                track.predict_miss()
                continue
                
            # 计算距离矩阵
            dists = [np.linalg.norm(track.center - d[0]) for d in unmatched_detections]
            min_idx = np.argmin(dists)
            
            # 如果距离小于阈值，说明是同一个苹果，更新它
            if dists[min_idx] < self.dist_thresh:
                track.update(unmatched_detections[min_idx][0], unmatched_detections[min_idx][1])
                unmatched_detections.pop(min_idx)
            else:
                track.predict_miss()
                
        # 2. 为没有匹配上的新检测创建新的跟踪器
        for d in unmatched_detections:
            self.tracks.append(TrackedApple(d[0], d[1]))
            
        # 3. 状态更新与销毁
        for track in self.tracks:
            if track.hits >= confirm_frames:
                track.is_confirmed = True
                
        # 销毁丢失太久的轨迹
        self.tracks = [t for t in self.tracks if t.misses <= max_lost_frames]
        
        # 4. 返回已确认存活的苹果
        return [(t.center, t.radius) for t in self.tracks if t.is_confirmed]

# ==================== 几何处理与拟合函数 ====================
def fit_sphere_least_squares(pts):
    if len(pts) < 15: return None, None
    A = np.zeros((len(pts), 4))
    A[:, 0:3] = pts
    A[:, 3] = 1
    B = np.sum(pts**2, axis=1)
    try:
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        center = res[0:3] / 2
        r2 = res[3] + np.sum(center**2)
        if r2 < 0: return None, None
        return center, np.sqrt(r2)
    except:
        return None, None

def segment_and_filter_apples(pts_3d, norm_angle_thresh, min_radius, max_radius):
    if len(pts_3d) < 50: return []
    red_pcd = o3d.geometry.PointCloud()
    red_pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    red_pcd = red_pcd.voxel_down_sample(voxel_size=0.005)
    if len(red_pcd.points) < 20: return []
    
    red_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    red_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    
    pts = np.asarray(red_pcd.points)
    normals = np.asarray(red_pcd.normals)
    
    cos_thresh = np.cos(np.radians(max(1, norm_angle_thresh))) 
    invalid_set = set()
    kdtree = o3d.geometry.KDTreeFlann(red_pcd)
    
    for i in range(len(pts)):
        if i in invalid_set: continue
        [k, idx, _] = kdtree.search_radius_vector_3d(pts[i], 0.012)
        if k > 1:
            dots = np.dot(normals[idx[1:]], normals[i])
            bad_local_idx = np.where(dots < cos_thresh)[0]
            if len(bad_local_idx) > 0:
                invalid_set.add(i)
                for bi in bad_local_idx:
                    invalid_set.add(idx[bi + 1])
                    
    valid_indices = [i for i in range(len(pts)) if i not in invalid_set]
    if len(valid_indices) < 20: return []
    filtered_pcd = red_pcd.select_by_index(valid_indices)
    
    labels = np.array(filtered_pcd.cluster_dbscan(eps=0.015, min_points=15))
    if len(labels) == 0 or labels.max() < 0: return []
    
    results = []
    for c in range(labels.max() + 1):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) < 20: continue
        
        cluster_pts = np.asarray(filtered_pcd.points)[cluster_idx]
        center, radius = fit_sphere_least_squares(cluster_pts)
        
        if center is not None and min_radius < radius < max_radius:
            results.append((center, radius))
            
    return results

# ==================== 2. FPS 飞行相机类 ====================
class FPSCamera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, -0.6]) 
        self.yaw, self.pitch = 0.0, 0.0
        self.look_at, self.right = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
        self.keys = set()
        self.mouse_sensitivity = 0.0012
        self.move_speed = 0.02
        self.lock_mouse, self.running = True, True
        self.last_apple_count = -1 

    def update_vectors(self):
        self.pitch = np.clip(self.pitch, -1.5, 1.5)
        front = np.array([np.cos(self.pitch) * np.sin(self.yaw), np.sin(self.pitch), np.cos(self.pitch) * np.cos(self.yaw)])
        self.look_at = front / np.linalg.norm(front)
        self.right = np.array([np.cos(self.yaw), 0, -np.sin(self.yaw)])

    def move(self):
        speed = self.move_speed * (5 if keyboard.Key.shift in self.keys else 1)
        if 'W' in self.keys: self.pos += self.look_at * speed
        if 'S' in self.keys: self.pos -= self.look_at * speed
        if 'A' in self.keys: self.pos += self.right * speed
        if 'D' in self.keys: self.pos -= self.right * speed
        if 'Q' in self.keys: self.pos[1] -= speed 
        if 'E' in self.keys: self.pos[1] += speed 

# ==================== 3. 初始化 Astra Mini S ====================
try:
    openni2_redist_path = "D:\\LIFE\\Summer\\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\\Win64-Release\\sdk\\libs"
    print("正在初始化 OpenNI2...")
    openni2.initialize(openni2_redist_path)
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    dev.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    depth_stream.start()
    color_stream.start()
except Exception as e:
    print(f"摄像头初始化失败: {e}")
    exit()

# ==================== 4. 输入监听 ====================
cam = FPSCamera()
mouse_ctrl = mouse.Controller()

def on_press(key):
    if key == keyboard.Key.esc: cam.running = False
    if key == keyboard.Key.tab: cam.lock_mouse = not cam.lock_mouse
    try: cam.keys.add(key.char.upper())
    except: cam.keys.add(key)
def on_release(key):
    try: cam.keys.discard(key.char.upper())
    except: cam.keys.discard(key)
k_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
k_listener.start()

# ==================== 5. 初始化 Open3D 与 OpenCV ====================
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Geometric Apple Tracker", width=1280, height=720)
ctr = vis.get_view_control()
try:
    ctr.set_constant_z_near(0.001); ctr.set_constant_z_far(100.0)
except: pass

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.zeros((307200, 3)))
pcd.colors = o3d.utility.Vector3dVector(np.zeros((307200, 3)))
vis.add_geometry(pcd)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))

MAX_APPLES = 5
apple_centers = []
apple_shells = []
for _ in range(MAX_APPLES):
    c = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    c.paint_uniform_color([0, 1, 0]) 
    c.translate([0, 0, -100]); vis.add_geometry(c)
    apple_centers.append(c)
    
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=12)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(s)
    ls.paint_uniform_color([1, 0.4, 0.4])
    ls.translate([0, 0, -100]); vis.add_geometry(ls)
    apple_shells.append(ls)

fx, fy, cx, cy = 575.0, 575.0, 312.0, 243.0
u, v = np.meshgrid(np.arange(640), np.arange(480))
u, v = u.flatten(), v.flatten()

cv2.namedWindow("Red Extraction", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Hue Tol", "Red Extraction", 10, 30, lambda x: None)
cv2.createTrackbar("Sat Min", "Red Extraction", 100, 255, lambda x: None)
cv2.createTrackbar("Val Min", "Red Extraction", 50, 255, lambda x: None)

cv2.createTrackbar("Depth Alpha", "Red Extraction", 60, 100, lambda x: None)

cv2.createTrackbar("Kernel", "Red Extraction", 5, 30, lambda x: None)
cv2.createTrackbar("Time Win", "Red Extraction", 3, 15, lambda x: None)   
cv2.createTrackbar("Norm Angle", "Red Extraction", 30, 90, lambda x: None) 
cv2.createTrackbar("Min Rad", "Red Extraction", 15, 100, lambda x: None)   
cv2.createTrackbar("Max Rad", "Red Extraction", 60, 200, lambda x: None)

# 【新增】：跟踪系统生命周期参数滑条
cv2.createTrackbar("Confirm Frm", "Red Extraction", 4, 20, lambda x: None) # 出现几帧后确认
cv2.createTrackbar("Lost Frm", "Red Extraction", 10, 30, lambda x: None)   # 丢失几帧后注销

# 实例化跟踪器
tracker = AppleTracker(dist_thresh=0.03) # 同一个苹果最大位移容忍度 10cm
mask_history = []
global_smooth_depth = None 

print("\n[控制说明]")
print("1. 新增跟踪系统：[Confirm Frm] 防止瞬间噪点；[Lost Frm] 防止短暂遮挡导致的消失。")
print("2. 配合 [Time Win] 使用，目前点云不仅在像素级稳定，且在 3D 对象级也具备了记忆能力！")

try:
    while cam.running:
        d_frame = depth_stream.read_frame()
        c_frame = color_stream.read_frame()
        d_arr_raw = np.frombuffer(d_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)
        c_arr = np.frombuffer(c_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
        bgr_img = cv2.cvtColor(c_arr, cv2.COLOR_RGB2BGR)

        # ---------------- 【核心】：深度平滑 EMA (指数移动平均) ----------------
        alpha_d = cv2.getTrackbarPos("Depth Alpha", "Red Extraction") / 100.0
        # 防止 alpha 为 0 导致深度不更新
        alpha_d = max(0.01, alpha_d) 

        # if global_smooth_depth is None:
        #     global_smooth_depth = d_arr_raw.astype(np.float32)
        # else:
        #     valid_mask = d_arr_raw > 0
            
        #     # 1. 以前是无效区域，现在变有效了，直接赋值初始化
        #     init_mask = (global_smooth_depth == 0) & valid_mask
        #     global_smooth_depth[init_mask] = d_arr_raw[init_mask].astype(np.float32)
            
            # # 2. 之前有效，现在也有效，执行 EMA 平滑混合
            # blend_mask = (global_smooth_depth > 0) & valid_mask
            # global_smooth_depth[blend_mask] = alpha_d * d_arr_raw[blend_mask] + (1.0 - alpha_d) * global_smooth_depth[blend_mask]

        # 得到平滑后的最终深度图使用
        # d_arr = global_smooth_depth.astype(np.uint16)
        d_arr=d_arr_raw.astype(np.float32)
        # -------------------------------------------------------------------------

        # ---------------- 环境更新 ----------------
        z = d_arr.flatten() / 1000.0
        x = (u - cx) * z / fx
        y = -(v - cy) * z / fy
        
        valid_z_mask = z > 0 
        pcd.points = o3d.utility.Vector3dVector(np.stack((x[valid_z_mask], y[valid_z_mask], z[valid_z_mask]), axis=-1))
        pcd.colors = o3d.utility.Vector3dVector((c_arr.reshape(-1, 3) / 255.0)[valid_z_mask])
        vis.update_geometry(pcd)

        # ---------------- 视角控制 ----------------
        if cam.lock_mouse:
            curr_x, curr_y = mouse_ctrl.position
            dx, dy = curr_x - CENTER_X, curr_y - CENTER_Y
            if dx != 0 or dy != 0:
                cam.yaw -= dx * cam.mouse_sensitivity
                cam.pitch -= dy * cam.mouse_sensitivity
                mouse_ctrl.position = (CENTER_X, CENTER_Y)
        cam.move(); cam.update_vectors()
        
        f_vec, r_vec = cam.look_at, cam.right
        up_vec = np.cross(r_vec, f_vec)
        R, T = np.eye(4), np.eye(4)
        R[0,:3], R[1,:3], R[2,:3] = r_vec, up_vec, f_vec
        T[:3, 3] = -cam.pos
        
        vp = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vp.extrinsic = R @ T
        vis.get_view_control().convert_from_pinhole_camera_parameters(vp, True)
        vis.poll_events(); vis.update_renderer()

        # ---------------- OpenCV 分割与累积 ----------------
        hue_tol = cv2.getTrackbarPos("Hue Tol", "Red Extraction")
        sat_min = cv2.getTrackbarPos("Sat Min", "Red Extraction")
        val_min = cv2.getTrackbarPos("Val Min", "Red Extraction")
        norm_angle = cv2.getTrackbarPos("Norm Angle", "Red Extraction")
        max_rad = cv2.getTrackbarPos("Max Rad", "Red Extraction") / 1000.0
        time_win = max(1, cv2.getTrackbarPos("Time Win", "Red Extraction"))
        
        confirm_f = max(1, cv2.getTrackbarPos("Confirm Frm", "Red Extraction"))
        lost_f = cv2.getTrackbarPos("Lost Frm", "Red Extraction")

        # 图像处理不需要太大的核，5x5 足够
        hsv_img = cv2.cvtColor(cv2.GaussianBlur(bgr_img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_img, np.array([0, sat_min, val_min]), np.array([hue_tol, 255, 255]))
        mask2 = cv2.inRange(hsv_img, np.array([180 - hue_tol, sat_min, val_min]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(red_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 300: cv2.drawContours(final_mask, [cnt], -1, 255, -1)

        mask_history.append(final_mask)
        if len(mask_history) > time_win: mask_history.pop(0)
        acc_mask = np.bitwise_or.reduce(mask_history) if len(mask_history) > 1 else mask_history[0]

        # ---------------- 几何处理与目标跟踪 ----------------
        y_idx, x_idx = np.where(acc_mask > 0)
        raw_detections = []
        if len(y_idx) > 0:
            z_v = d_arr[y_idx, x_idx] / 1000.0
            valid_obj = (z_v > 0.1) & (z_v < 2.5) 
            if np.any(valid_obj):
                z_v = z_v[valid_obj]
                x_v = (x_idx[valid_obj] - cx) * z_v / fx
                y_v = -(y_idx[valid_obj] - cy) * z_v / fy
                red_pts_3d = np.stack((x_v, y_v, z_v), axis=-1)
                
                # Minimum radius fixed to 1.5cm for apple logic
                raw_detections = segment_and_filter_apples(red_pts_3d, norm_angle, 0.015, max_rad)
        
        confirmed_apples = tracker.update(raw_detections, confirm_f, lost_f)
        
        if len(confirmed_apples) != cam.last_apple_count:
            print(f"-> 视觉分析：当前锁定并追踪了 {len(confirmed_apples)} 个苹果")
            cam.last_apple_count = len(confirmed_apples)

        # ---------------- 多目标渲染 ----------------
        for i in range(MAX_APPLES):
            if i < len(confirmed_apples):
                center, radius = confirmed_apples[i]
                apple_centers[i].translate(center, relative=False)
                
                temp_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
                temp_ls = o3d.geometry.LineSet.create_from_triangle_mesh(temp_mesh)
                temp_ls.translate(center, relative=False)
                
                apple_shells[i].points = temp_ls.points
                apple_shells[i].lines = temp_ls.lines
                apple_shells[i].colors = temp_ls.colors
                apple_shells[i].paint_uniform_color([1, 0.4, 0.4]) 
            else:
                apple_centers[i].translate([0, 0, -100], relative=False)
                apple_shells[i].points = o3d.utility.Vector3dVector()
                apple_shells[i].lines = o3d.utility.Vector2iVector()
            
            vis.update_geometry(apple_centers[i])
            vis.update_geometry(apple_shells[i])

        # ---------------- OpenCV UI ----------------
        red_extracted_img = cv2.bitwise_and(bgr_img, bgr_img, mask=acc_mask)
        MAX_DEPTH_MM = 3000.0  
        depth_colormap = cv2.applyColorMap((np.clip(d_arr, 0, MAX_DEPTH_MM) / MAX_DEPTH_MM * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        depth_colormap[d_arr == 0] = [0, 0, 0]

        cv2.imshow("Astra RGB", bgr_img)
        cv2.imshow("Astra Depth", depth_colormap) 
        cv2.imshow("Red Extraction", red_extracted_img) 
        
        if cv2.waitKey(1) == 27: break

finally:
    print("\n程序关闭，释放鼠标...")
    k_listener.stop()
    depth_stream.stop()
    color_stream.stop()
    dev.close()
    openni2.unload()
    vis.destroy_window()
    cv2.destroyAllWindows()
