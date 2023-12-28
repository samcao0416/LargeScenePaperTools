import open3d as o3d
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str, required=True, help='input point cloud path')
parser.add_argument('-out', '--output', type=str, default=None, help='output point cloud path')
args = parser.parse_args()

in_path = args.input
out_path = args.output
if out_path is None:
    out_path = os.path.join(os.path.dirname(in_path), os.path.basename(in_path)[:-4] + "_out.ply")
# 1. 读取ply格式的点云
pcd = o3d.io.read_point_cloud(in_path)

# 2. 获取点云的边界信息，并扩大 50%
bounds = pcd.get_axis_aligned_bounding_box()
min_bound = bounds.get_min_bound()
max_bound = bounds.get_max_bound()
bound_dist = max_bound - min_bound
min_bound -= bound_dist * 0.5
max_bound += bound_dist * 0.5
x_min, y_min, z_min = min_bound
x_max, y_max, z_max = max_bound

# 3. 在x_min到x_max和y_min到y_max这个距离内，每隔0.5增加一个白色的点
dist = 1.0
x_range = np.arange(x_min, x_max, dist)
y_range = np.arange(y_min, y_max, dist)
z_range = np.array([z_max])  # 只包含z_max


# 创建顶面的点云
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
top_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

x_range = np.arange(x_min, x_max, dist)
y_range = np.arange(y_min, y_max, dist)
z_range = np.array([z_min])  # 只包含z_max


# 创建顶面的点云
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
bot_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# 创建四个侧面的点云
side_points = []

x_range = np.array([x_max])
y_range = np.arange(y_min, y_max, dist)
z_range = np.arange(z_min, z_max, dist)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
side_points.extend(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)

x_range = np.array([x_min])
y_range = np.arange(y_min, y_max, dist)
z_range = np.arange(z_min, z_max, dist)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
side_points.extend(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)

x_range = np.arange(x_min, x_max, dist)
y_range = np.array([y_max])
z_range = np.arange(z_min, z_max, dist)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
side_points.extend(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)

x_range = np.arange(x_min, x_max, dist)
y_range = np.array([y_min])
z_range = np.arange(z_min, z_max, dist)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
side_points.extend(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T)

# 合并顶面和侧面的点云
new_points = np.vstack((top_points, side_points, bot_points))
new_colors = np.ones((new_points.shape[0], 3))  # 白色

# 创建Open3D点云对象
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_points)
new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

# 合并原始点云和新创建的点云
merged_pcd = pcd + new_pcd

# 可视化
o3d.io.write_point_cloud(out_path, merged_pcd)

print("Ply saved to: ", out_path)