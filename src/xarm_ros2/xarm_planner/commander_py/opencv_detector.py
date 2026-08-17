#!/usr/bin/env python3
"""
OpenCV 多特征物体识别 + ROI 点云中值 + 卡尔曼滤波
颜色 + 几何 + 边缘一致性过滤, 点云仅读 ROI, 深度方差检查
输出 TF + PointStamped: obj_green, obj_red, obj_blue
"""

from collections import deque
import cv2, numpy as np, rclpy, tf2_ros, time
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


COLORS = {
    "green": {"hsv_lo":[38,40,40],"hsv_hi":[82,255,255],"bgr":(0,255,0),"tf":"obj_green",
              "min_area":200,"shape":"circle","shape_thresh":0.55},
    "red":   {"hsv_lo":[0,40,40],"hsv_hi":[10,255,255],"hsv_lo2":[170,40,40],"hsv_hi2":[180,255,255],
              "bgr":(0,0,255),"tf":"obj_red","min_area":200,"shape":"rect","shape_thresh":0.60},
    "blue":  {"hsv_lo":[95,40,40],"hsv_hi":[135,255,255],"bgr":(255,0,0),"tf":"obj_blue",
              "min_area":200,"shape":"rect","shape_thresh":0.60},
}

SMOOTH_WINDOW = 5
JUMP_THRESH = 0.10
MAX_ASPECT_RATIO = 4.0
MIN_EDGE_RATIO = 0.01
MAX_EDGE_RATIO = 0.45
ROI_HALF = 5
MIN_VALID_POINTS = 8
DEPTH_MIN = 0.03
DEPTH_MAX = 3.0
DEPTH_VAR_MAX = 0.05  # 深度方差上限（放宽避免真实场景误剔除）


class Kalman1D:
    """简单 1D 卡尔曼 (x,vx)"""
    def __init__(self):
        self.x = np.array([0.,0.])
        self.P = np.eye(2)*0.1
        self.F = np.array([[1.,1.],[0.,1.]])
        self.H = np.array([[1.,0.]])
        self.Q = np.eye(2)*0.001
        self.R = np.array([[0.005]])
        self._init = False

    def update(self, z):
        if not self._init:
            self.x[0] = z
            self._init = True
            return z
        xp = self.F @ self.x; Pp = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ xp; S = self.H @ Pp @ self.H.T + self.R
        K = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ y
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return self.x[0]


class ColorDetector(Node):
    def __init__(self):
        super().__init__("color_detector")
        self.declare_parameter('depth_var_max', DEPTH_VAR_MAX)
        self.declare_parameter('min_area', 300)
        self.declare_parameter('jump_thresh', JUMP_THRESH)

        self._depth_var_max = self.get_parameter('depth_var_max').value
        self._min_area = self.get_parameter('min_area').value
        self._jump_thresh = self.get_parameter('jump_thresh').value
        self.bridge = CvBridge()
        self._tf = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(Image, "/camera/color/image_raw", self._cb_rgb, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self._cb_cinfo, 10)
        self.create_subscription(PointCloud2, "/camera/depth/points", self._cb_pc, 10)
        self._pub = self.create_publisher(Image, "/color_detector/debug", 10)
        self._pos_pubs = {c["tf"]: self.create_publisher(PointStamped, "/"+c["tf"]+"_pos", 10) for c in COLORS.values()}
        self._rgb = self._pc = None; self._fx = self._fy = self._cx = self._cy = 0.0
        self._kalman = {cfg["tf"]: [Kalman1D() for _ in range(3)] for cfg in COLORS.values()}
        self._history = {cfg["tf"]: deque(maxlen=SMOOTH_WINDOW) for cfg in COLORS.values()}
        self._pc_arr = None; self._pc_shape = (0,0); self._pc_frame = ""
        self._pc_frame_logged = False
        self._xyz_logged = False
        self._pc_indices = None
        self._fps_t = time.time(); self._fps_count = 0; self._fps_val = 0.0
        self.get_logger().info("ColorDetector v2 (shape-specific + kalman + depth variance)")

    # ---- callbacks ----
    def _cb_cinfo(self, m): self._fx,self._fy,self._cx,self._cy = m.k[0],m.k[4],m.k[2],m.k[5]
    def _cb_rgb(self, m): self._rgb = self.bridge.imgmsg_to_cv2(m, "bgr8")
    def _cb_pc(self, m):
        self._pc = m
        h,w = m.height, m.width
        self._pc_frame = m.header.frame_id
        if not self._pc_frame_logged:
            self.get_logger().info(f"PointCloud2 frame: {self._pc_frame}")
            self._pc_frame_logged = True
        fields = {field.name: field.offset // 4 for field in m.fields}
        if not all(name in fields for name in ("x", "y", "z")):
            self.get_logger().error("PointCloud2 has no x/y/z fields")
            return
        self._pc_indices = (fields["x"], fields["y"], fields["z"])
        words_per_point = m.point_step // 4
        words_per_row = m.row_step // 4
        data = np.frombuffer(m.data, dtype=np.float32).reshape(h, words_per_row)
        self._pc_arr = data[:, :w * words_per_point].reshape(h, w, words_per_point)
        self._pc_shape = (h,w)
        if self._rgb is not None and self._fx != 0:
            self._detect()

    def _point_to_optical(self, point):
        """Convert Gazebo's camera-depth frame to the ROS optical frame."""
        x, y, z = point
        if self._pc_frame.endswith("cameradepth"):
            # Gazebo camera sensor convention is x-forward, y-left, z-up.
            # ROS optical convention is x-right, y-down, z-forward.
            return (-y, -z, x)
        return (x, y, z)

    # ---- 颜色掩膜 ----
    def _make_color_mask(self, hsv, cfg):
        mask = cv2.inRange(hsv, np.array(cfg["hsv_lo"]), np.array(cfg["hsv_hi"]))
        if "hsv_lo2" in cfg:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(cfg["hsv_lo2"]), np.array(cfg["hsv_hi2"])))
        k = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # ---- 轮廓特征 ----
    def _features(self, c):
        a = cv2.contourArea(c); x,y,w,h = cv2.boundingRect(c)
        ra = max(w*h,1); p = cv2.arcLength(c, True)
        return {"area":a,"bbox":(x,y,w,h),"aspect":max(w/max(h,1),h/max(w,1)),
                "extent":a/ra,"circ":0.0 if p<=0 else 4*np.pi*a/(p*p)}

    def _shape_score(self, c, shape_type):
        a = cv2.contourArea(c); p = cv2.arcLength(c, True)
        if shape_type == "circle": return 0.0 if p<=0 else 4*np.pi*a/(p*p)
        if shape_type == "rect":
            rect = cv2.minAreaRect(c)
            ba = rect[1][0]*rect[1][1]
            return a/max(ba,1) if ba>0 else 0.0
        return 1.0

    # ---- 边缘一致性 ----
    def _edge_ratio(self, gray, c, bbox):
        x,y,w,h = bbox; roi = gray[y:y+h, x:x+w]
        if roi.size==0: return 0.0
        rm = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(rm, [c-np.array([[[x,y]]])], -1, 255, -1)
        edges = cv2.Canny(roi, 60, 160)
        return cv2.countNonZero(cv2.bitwise_and(edges,rm))/max(cv2.countNonZero(rm),1)

    # ---- 候选验证 ----
    def _is_valid(self, c, cfg, gray):
        f = self._features(c)
        if f["area"]<cfg["min_area"] or f["aspect"]>MAX_ASPECT_RATIO:
            return False,f
        shape = self._shape_score(c, cfg["shape"])
        if shape < cfg["shape_thresh"]:
            return False,f
        f["edge_ratio"] = self._edge_ratio(gray, c, f["bbox"])
        if not (MIN_EDGE_RATIO<=f["edge_ratio"]<=MAX_EDGE_RATIO):
            return False,f
        f["shape"] = shape
        return True,f

    # ---- 评分 ----
    def _score(self, f):
        return (0.35*min(f["area"]/3000,1.0)+0.20*min(f["extent"],1.0)+
                0.25*min(f["shape"],1.0)+0.20*(1-min(abs(f["edge_ratio"]-0.1)/0.1,1.0)))

    # ---- ROI 点云 (仅采样轮廓内) + 深度方差检查 ----
    def _get_xyz_roi(self, c, cx, cy):
        if self._pc_arr is None or self._pc_indices is None: return None
        h,w = self._pc_shape
        pts = []
        raw_pts = []
        for py in range(max(0,cy-ROI_HALF), min(h,cy+ROI_HALF)):
            for px in range(max(0,cx-ROI_HALF), min(w,cx+ROI_HALF)):
                if cv2.pointPolygonTest(c, (px,py), False)>=0:
                    raw = self._pc_arr[py, px]
                    raw_xyz = (
                        raw[self._pc_indices[0]],
                        raw[self._pc_indices[1]],
                        raw[self._pc_indices[2]],
                    )
                    raw_pts.append(tuple(float(v) for v in raw_xyz))
                    x, y, z = self._point_to_optical(raw_xyz)
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and DEPTH_MIN<z<DEPTH_MAX:
                        pts.append((float(x),float(y),float(z)))
        if len(pts)<MIN_VALID_POINTS: return None
        arr = np.array(pts)
        # 深度方差检查
        if np.var(arr[:,2]) > self._depth_var_max: return None
        result = tuple(np.median(arr, axis=0))
        if not self._xyz_logged:
            raw_median = tuple(np.median(np.array(raw_pts), axis=0))
            self.get_logger().info(
                f"XYZ check frame={self._pc_frame} raw={raw_median} optical={result}")
            self._xyz_logged = True
        return result

    # ---- 时序平滑 (滑动平均 + 跳变检测) ----
    def _smooth(self, tf_name, xyz):
        buf = self._history[tf_name]
        if buf and xyz and np.linalg.norm(np.array(xyz)-np.array(buf[-1]))>self._jump_thresh:
            buf.clear()
        if xyz: buf.append(xyz)
        return tuple(np.mean(buf, axis=0)) if buf else None

    # ---- 发送 ----
    def _publish(self, tf_name, xyz, f, name, debug, cx, cy, cfg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_depth_optical_frame"
        t.child_frame_id = tf_name
        t.transform.translation.x = xyz[0]; t.transform.translation.y = xyz[1]; t.transform.translation.z = xyz[2]
        t.transform.rotation.w = 1.0
        self._tf.sendTransform(t)
        ps = PointStamped(); ps.header.stamp = t.header.stamp
        ps.header.frame_id = "camera_depth_optical_frame"
        ps.point.x = xyz[0]; ps.point.y = xyz[1]; ps.point.z = xyz[2]
        self._pos_pubs[tf_name].publish(ps)
        xb,yb,wb,hb = f["bbox"]
        cv2.rectangle(debug, (xb,yb), (xb+wb,yb+hb), cfg["bgr"], 2)
        cv2.circle(debug, (cx,cy), 4, cfg["bgr"], -1)
        cv2.putText(debug, f"{name} z={xyz[2]:.2f} s={f['shape']:.2f} e={f['edge_ratio']:.2f}",
                    (xb, max(yb-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg["bgr"], 2)

    # ---- 主检测 ----
    def _detect(self):
        img = self._rgb
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        debug = img.copy()

        for name, cfg in COLORS.items():
            mask = self._make_color_mask(hsv, cfg)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue

            candidates = []
            for c in contours:
                ok, f = self._is_valid(c, cfg, gray)
                if ok: candidates.append((self._score(f), c, f))
            if not candidates: continue

            # 输出最多 2 个候选 (多物体)
            candidates.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, c, f) in enumerate(candidates[:2]):
                M = cv2.moments(c)
                if M["m00"]==0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                xyz = self._get_xyz_roi(c, cx, cy)
                if xyz is None: continue
                xyz = self._smooth(cfg["tf"], xyz)
                if xyz is None: continue
                tf_suffix = f"_{rank}" if rank>0 else ""
                self._publish(cfg["tf"]+tf_suffix, xyz, f, name, debug, cx, cy, cfg)

        self._fps_count += 1
        t_now = time.time()
        if t_now - self._fps_t >= 1.0:
            self._fps_val = self._fps_count / (t_now - self._fps_t)
            self._fps_t = t_now; self._fps_count = 0
        cv2.putText(debug, f"FPS: {self._fps_val:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        self._pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))


def main():
    rclpy.init(); rclpy.spin(ColorDetector()); rclpy.shutdown()

if __name__=="__main__":
    main()
