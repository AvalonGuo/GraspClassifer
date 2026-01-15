def compute_mujoco_camera_params(
    x_range,
    y_range,
    camera_z,
    object_z,
    pixel_size=0.001,
    focal=0.01,
    camera_name="pick_view"
):
    """
    自动计算 MuJoCo 相机参数，使视野刚好覆盖指定区域，并满足像素精度。
    
    默认相机位于目标区域中心正上方，使用对称焦距 (fx=fy)。
    
    参数:
    ------
    x_range : tuple (xmin, xmax) —— 目标 x 范围（米）
    y_range : tuple (ymin, ymax) —— 目标 y 范围（米）
    camera_z : float             —— 相机 z 坐标（米）
    object_z : float             —— 成像平面 z 坐标（米）
    pixel_size : float           —— 每像素对应多少米（默认 0.001 = 1mm/pixel）
    focal : float                —— 焦距（米），默认 0.01（10mm）
    camera_name : str            —— 相机名称（用于 XML）

    返回:
    ------
    dict: 包含 'xml_snippet' 及其他参数
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    Wx = xmax - xmin
    Wy = ymax - ymin
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    d = camera_z - object_z
    if d <= 0:
        raise ValueError("错误：相机必须在成像平面上方（camera_z > object_z）")

    # 分辨率由物理尺寸和像素精度决定
    res_x = round(Wx / pixel_size)
    res_y = round(Wy / pixel_size)
    if res_x < 1 or res_y < 1:
        raise ValueError("分辨率太小，请增大 pixel_size 或扩大视野范围")

    # 计算传感器尺寸
    sx = (focal / d) * Wx
    sy = (focal / d) * Wy

    # 实际像素精度（应≈输入值）
    actual_px = Wx / res_x
    actual_py = Wy / res_y

    xml_snippet = f"""<body pos="{center_x} {center_y} {camera_z}">
    <camera name="{camera_name}"
            mode="fixed"
            focal="{focal} {focal}"
            sensorsize="{sx:.6f} {sy:.6f}"
            resolution="{res_x} {res_y}"/>
</body>"""

    return {
        "camera_pos": (center_x, center_y, camera_z),
        "focal": (focal, focal),
        "sensorsize": (sx, sy),
        "resolution": (res_x, res_y),
        "pixel_size_actual": (actual_px, actual_py),
        "xml_snippet": xml_snippet
    }

# 主平台范围
x_range = (-0.28, 0.28)
y_range = (0.4, 0.8)
# place : <!--x ∈ [0.4, 0.8]   y ∈ [-0.2, 0.4] -->
# x_range = (0.4,0.8)
# y_range = (-0.2,0.4)
# 相机高度和平台高度
camera_z = 0.765
object_z = -0.003

# 一键生成（全部用默认值：1mm/pixel, focal=0.01）
params = compute_mujoco_camera_params(x_range, y_range, camera_z, object_z)

print(params["xml_snippet"])