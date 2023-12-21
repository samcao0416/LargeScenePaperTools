import copy
import sys
import open3d as o3d
import numpy as np
from scipy.spatial.transform.rotation import Rotation
import glob
import os

def extrinsic_matrix(q, trans_vec):
    ext_mat = np.eye(4)
    R = Rotation.as_matrix(Rotation.from_quat(q))
    ext_mat[:3, :3] = R
    ext_mat[:3, -1] = trans_vec
    return ext_mat

def create_rays(pcd, trans_vec):
    # Create rays from the trans_vec point to each point in the point cloud
    rays = np.zeros((len(pcd.points), 6))
    rays[:, :3] = trans_vec # origin is the same as the trans_vec
    rays[:, 3:] = pcd.points - trans_vec # direction is the vector from trans_vec to point
    return o3d.core.Tensor(rays)
from scipy.interpolate import RegularGridInterpolator

def find_closest_ply(timestamp,path):
    pcd_files = sorted(glob.glob(path + "/*.ply"))
    min_diff = float("inf")
    closest_file = None
    for pcd_file in pcd_files:
        #file_timestamp = float(os.path.basename(pcd_file).split(".")[0])/1000000
        # if "ouster" in pcd_file:
        file_timestamp = float(os.path.basename(pcd_file).split(".")[0])+float(os.path.basename(pcd_file).split(".")[1])/1000000
        # print(pcd_files,file_timestamp)
        diff = abs(file_timestamp - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_file = pcd_file
    if min_diff>0.1:
        print("the min_diff is too large",min_diff,timestamp,closest_file)
    print(closest_file, timestamp)
    return closest_file

def write_point_cloud(filename, pcd):
    # Open the file in write mode
    with open(filename, "w") as f:
        # Write the header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pcd.points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        #rgb
        # f.write("property uchar red\n")
        # f.write("property uchar green\n")
        # f.write("property uchar blue\n")
        # normal
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property float intensity\n")
        f.write("property float ring\n")
        # f.write("property float index\n")
        f.write("property float return\n")
        # f.write("property float depth\n")
        f.write("property float label\n")
        f.write("end_header\n")

        # Write each point as a line
        for i in range(len(pcd.points)):
            x, y, z = pcd.points[i]
            #rgb 0-255
            # r = int(pcd.colors[i][0]*255)
            # g = int(pcd.colors[i][1]*255)
            # b = int(pcd.colors[i][2]*255)
            # normal
            nx = pcd.normals[i][0]
            ny = pcd.normals[i][1]
            nz = pcd.normals[i][2]
            intensity = pcd.covariances[i][0][0]
            ring = pcd.covariances[i][0][1]
            # index = pcd.covariances[i][0][2]
            return_ = pcd.covariances[i][1][0]
            # depth = pcd.covariances[i][2][0]
            label = pcd.covariances[i][2][1]
            # f.write(f"{x} {y} {z} {r} {g} {b} {nx} {ny} {nz} {intensity} {ring} {index} {return_} {depth} {label}\n")
            f.write(f"{x} {y} {z} {nx} {ny} {nz} {intensity} {ring} {return_} {label}\n")

    print(f"Point cloud saved to {filename}")

def read_point_cloud(filename):
    # Read the header of the ply file
    with open(filename, "r") as f:
        header = f.readline().strip()
        # Check if the file is in ascii format
        if header != "ply":
            print("The file is not in ply format")
            return None
        # Skip the next line of the header
        f.readline()
        # Read the number of vertices from the header
        num_vertices = int(f.readline().split()[2])
        # Skip the next 10 lines of the header
        for _ in range(8):
            f.readline()
        # Read the data as a numpy array
        data = np.loadtxt(f)
    # Create an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    # Set the xyz coordinates from the data
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    # Set the covariances from the data
    covariances = np.zeros((num_vertices, 3, 3))
    covariances[:, 0, 0] = data[:, 3] # intensity
    covariances[:, 0, 1] = data[:, 4] # ring
    covariances[:, 0, 2] = data[:, 5] # index
    covariances[:, 1, 0] = data[:, 6] # return
    # pcd.covariances[:, 2, 0] = data[:, 7] # depth
    # pcd.covariances[:, 2, 1] = data[:, 8] # label
    pcd.covariances = o3d.utility.Matrix3dVector(covariances)
    return pcd

def interp_texture_map_from_ray_tracing_results(mesh, texture, raycast_result):
  # Get the primitive ids and barycentric coordinates from the raycast result
  prim_ids = raycast_result['primitive_ids'].numpy()
  uvs = raycast_result['primitive_uvs'].numpy()

  # Get the texture coordinates of the mesh
  tex_coords = np.asarray(mesh.triangle_uvs)

  # Initialize an array to store the color values
  color = np.zeros((len(prim_ids), 3))

  # Loop over each ray
  for i in range(len(prim_ids)):
    # Get the primitive id of the hit triangle
    prim_id = prim_ids[i]

    # Skip if the ray did not hit any triangle
    if prim_id == -1 or prim_id >= len(tex_coords) // 3:
      continue

    # Get the barycentric coordinates of the hit point
    u = uvs[i, 0]
    v = uvs[i, 1]
    w = 1 - u - v
    # print(prim_id)
    # Get the texture coordinates of the three vertices of the triangle
    tex_coord0 = tex_coords[3 * prim_id]
    tex_coord1 = tex_coords[3 * prim_id + 1]
    tex_coord2 = tex_coords[3 * prim_id + 2]

    # Interpolate the texture coordinates using the barycentric coordinates
    tex_coord = u * tex_coord0 + v * tex_coord1 + w * tex_coord2

    # Get the texture pixel coordinates by multiplying with the texture size
    tex_x = int(tex_coord[0] * texture.shape[1])
    tex_y = int(tex_coord[1] * texture.shape[0])
    # if > 4095, set to 4095
    if tex_x > 4095:
        tex_x = 4095
    if tex_y > 4095:
        tex_y = 4095
    # Get the color value from the texture image
    tex_color = texture[tex_y, tex_x]

    # Store the color value in the output array
    color[i] = tex_color

  # Return the color array
  return color
"""
  // cast rays to get uv
  std::vector<std::array<float, 3>> vVertices(vertices->points.size());
  #pragma omp parallel for
  for (uint i = 0; i < vertices->points.size(); ++i) {
    auto &p = vertices->points[i];
    vVertices[i] = {p.x, p.y, p.z};
  }
  std::vector<uint32_t> vTriangles(meshTexture.tex_polygons[0].size()*3);
  #pragma omp parallel for
  for (uint i = 0; i < meshTexture.tex_polygons[0].size(); ++i) {
    vTriangles[3*i] = meshTexture.tex_polygons[0][i].vertices[0];
    vTriangles[3*i+1] = meshTexture.tex_polygons[0][i].vertices[1];
    vTriangles[3*i+2] = meshTexture.tex_polygons[0][i].vertices[2];
  }
  Raycast rc;
  rc.addTriangles(vVertices, vTriangles);
  auto raydists = rc.castRays(mvRays);
  cv::Mat single(mPanoHeight, mPanoWidth, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat alpha(single.size(), CV_8UC1, cv::Scalar(0));
  for (int i = 0; i < mvRays.size(); ++i) {
    auto dist = std::get<0>(raydists)[i];
    if (std::isinf(dist)) continue;
    auto faceId = std::get<1>(raydists)[i];
    if (faceId == RTC_INVALID_GEOMETRY_ID) continue;
    float &rayU = std::get<2>(raydists)[2*i];
    float &rayV = std::get<2>(raydists)[2*i+1];
    // see https://community.intel.com/t5/Intel-Embree-Ray-Tracing-Kernels/Barycentric-coordinates-u-v/td-p/1016800
    Eigen::Vector2f uv = (1.0f-rayU-rayV)*meshTexture.tex_coordinates[0][3*faceId+0]
                          + rayU*meshTexture.tex_coordinates[0][3*faceId+1]
                          + rayV*meshTexture.tex_coordinates[0][3*faceId+2];
    // Note that uv coord is different from cv coord!!!
    auto rgb = undistorted.at<cv::Vec3b>((1.0f-uv(1))*float(undistorted.rows), uv(0)*float(undistorted.cols));
    int oriRow = i / single.cols;
    int oriCol = i % single.cols;
    single.at<cv::Vec3b>(oriRow, oriCol) = rgb;
    alpha.at<uint8_t>(oriRow, oriCol) = 255;
  }
  """


def raycast_point(originpcd, ext_mat):
    pcd = copy.deepcopy(originpcd)
    pcd.transform(ext_mat)

    # get trans_vec from ext_mat
    trans_vec = ext_mat[0:3, 3]
    #print(trans_vec)
    # Create rays from the origin point to each point in the point cloud
    rays = create_rays(pcd, trans_vec)
    rays = rays.to(o3d.core.Dtype.Float32)
    # Raycast to the mesh and get the depth and texture uv in the mesh
    raycast_result = scene.cast_rays(rays)
    depth = raycast_result['t_hit'].numpy()
    # Get the color from the mesh using the primitive uvs
    texture = np.asarray(mesh.textures[0]) / 255.  # (h,w,3)`
    color = interp_texture_map_from_ray_tracing_results(mesh, texture, raycast_result)
    # Convert the depth array to a numpy array
    color = np.asarray(color)
    pcd.colors = o3d.utility.Vector3dVector(color)
    # visualize the pointcloud
    # o3d.visualization.draw_geometries([mesh,pcd])
    normal = raycast_result['primitive_normals'].numpy()
    originpcd.normals = o3d.utility.Vector3dVector(normal)
    depth = np.asarray(depth)
    # if depth is inf, set it to 10000
    depth = np.where(depth == np.inf, 10000, depth)
    covariances = np.asarray(originpcd.covariances)
    covariances[:, 2, 0] = depth
    # fill the label with -1
    label = np.zeros(len(pcd.points)) 

 
    mask_raycasttoglass = (color[:, 1] == 1) & (color[:, 0] == 0) & (color[:, 2] == 0)
    mask_raycasttomirror  = (color[:, 1] == 0) & (color[:, 0] == 0) & (color[:, 2] == 1)
    mask_raycasttootherreflect = (color[:, 1] == 1) & (color[:, 0] == 0) & (color[:, 2] == 1)
    # intensity is low and distance are near depth
    distance = np.linalg.norm(pcd.points - trans_vec, axis=1)
    mesh_minus_point = distance * (depth - 1)
    mask_distancenear = np.abs(mesh_minus_point) < 0.15
    # # debug print distance and depth for some points
    # print(rays[1000:1010])
    # print((pcd.points - trans_vec)[1000:1010])
    # print(((pcd.points - trans_vec) * depth.reshape(-1, 1))[1000:1010])
    #
    # distance = np.linalg.norm(pcd.points - trans_vec, axis=1)
    #
    # depth_normal = ((pcd.points - trans_vec) * depth.reshape(-1, 1))
    # depth_distance = np.linalg.norm(depth_normal, axis=1)
    #
    # print(np.abs(distance - depth_distance))

    # # covert depth to new pointcloud and visualize for debug
    # pcd_depth = o3d.geometry.PointCloud()
    # # point cloud from depth time depth with ray to get the point
    # pcd_depth.points = o3d.utility.Vector3dVector((pcd.points - trans_vec) * depth.reshape(-1, 1))
    # print(len(pcd_depth.points))
    # o3d.visualization.draw_geometries([pcd_depth])

    #distance * depth + 0.07  < distance : outsidemesh
    mask_outsidemesh = mesh_minus_point < -0.15
    # remove inf (depth 10000) from mask_outsidemesh
    mask_outsidemesh = mask_outsidemesh & (depth < 100 )
    #distance * depth < distance + 0.07 : outsidemesh
    # distance more than 20m
    mask_distancefar = distance > 30
    # mask_insidemesh = mesh_minus_point > 0.15

    #label depth not inf as label 1, normal point
    label[(depth < 100) & ~mask_raycasttoglass & ~mask_raycasttomirror & ~mask_raycasttootherreflect & ~mask_distancefar] = 1
    # glass point label 2, mirror point label 3, other reflection point label 4
    label[mask_raycasttoglass & mask_distancenear] = 2
    label[mask_raycasttomirror & mask_distancenear] = 3
    label[mask_raycasttootherreflect & mask_distancenear] = 4
    label[mask_raycasttomirror & mask_outsidemesh] = 5
    label[mask_raycasttootherreflect & mask_outsidemesh] = 5

    # label[mask_insidemesh] = 5
    # label[~mask_isglass & ~mask_ismirror & mask_distancenear] = 0
    # label[~mask_isglass & ~mask_ismirror & mask_outsidemesh] = 4
    # label[mask_ismirror & mask_distancenear] = 6
    # label to color
    # label normal points as 0, glass points as 1, and reflection points as 2, outside obstacle points as 3, outside noise points as 4, inside noise points as 5, mirror points as 6, out of range points as -1

    # get the points outside the green area
    label[mask_raycasttoglass & mask_distancefar] = 6
    pcd_points = np.asarray(pcd.points)
    points_outside_green = pcd_points[mask_raycasttoglass & mask_outsidemesh & ~mask_distancefar]
    # label_outside_green = label[mask_isglass & mask_outsidemesh]
    # calculate the distance between the points and the mesh
    distance_outside = scene.compute_distance(points_outside_green.astype(np.float32)).numpy()
    # if the distance is less than 0.07, then the point is a reflection point, else it is an outside obstacle point
    reflect_mask = distance_outside > 0.15
    # get the indices of the points that are outside the green area
    indices = np.where(mask_raycasttoglass & mask_outsidemesh & ~mask_distancefar)[0]
    # update the label array using those indices
    label[np.ravel(indices[reflect_mask])] = 5
    label[np.ravel(indices[~reflect_mask])] = 6

    # label_color = np.zeros((len(pcd.points), 3))
    # label_color[label == 0] = [0, 0, 0] # black, unlabeled
    # label_color[label == 1] = [0.5, 0.5,0.5] # grey, normal
    # label_color[label == 2] = [0, 1, 0] # green, glass
    # label_color[label == 3] = [0, 0, 1] # blue，mirror
    # label_color[label == 4] = [0, 1, 1] # cyan，other reflective
    # label_color[label == 5] = [1, 0, 1] # magenta，reflection
    # label_color[label == 6] = [1, 1, 0] # yellow，outside obstacle
    # pcd.colors = o3d.utility.Vector3dVector(label_color)
    # o3d.visualization.draw_geometries([pcd])

    # put the label to origin pcd

    # debug print distance and depth for some points
    # put the distance into the cloud and save it
    # testcloud = o3d.geometry.PointCloud()
    # testcloud.points = o3d.utility.Vector3dVector(pcd_points[mask_isgreen & mask_outsidemesh])
    # o3d.visualization.draw_geometries([testcloud])

    # covariancestest = np.zeros((len(testcloud.points), 3, 3))
    # covariancestest[:, 2, 0] = distance_outside.numpy()
    # testcloud.covariances = o3d.utility.Matrix3dVector(covariancestest)
    # testcloud.paint_uniform_color([0, 1, 0])
    # testcloud.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask_isgreen & mask_outsidemesh])
    # write_point_cloud("test.ply", testcloud)
    # exit(0)
    # Householder transformation

    # # get the normal of the glass
    # reflect_mask_new = np.ravel(indices[reflect_mask])
    # reflect_points = pcd_points[reflect_mask_new]
    # plane_normal = normal[reflect_mask_new]
    # ray_direction = reflect_points - trans_vec
    # # get the depth of the plane
    # plane_depth = depth[reflect_mask_new]
    # # get any point on the plane by scaling the ray direction with the depth
    # point_on_plane = trans_vec + ray_direction * plane_depth[:, None]
    # # calculate d
    # # convert point_on_plane to pointcloud and visualize for debug
    # cloud_point_on_plane = o3d.geometry.PointCloud()
    # cloud_point_on_plane.points = o3d.utility.Vector3dVector(point_on_plane)
    # reflect_points_cloud = o3d.geometry.PointCloud()
    # reflect_points_cloud.points = o3d.utility.Vector3dVector(reflect_points)
    # d = -np.sum(plane_normal * point_on_plane, axis=1)
    # # mirror the point use Householder transformation
    # transform = np.eye(3) - (2 * plane_normal[:, :, None] * plane_normal[:, None, :])
    # affine = transform @ reflect_points[:, :, None]
    # affine += -2 * plane_normal[:, :, None] * d[:, None, None]
    # mirror_point = affine.squeeze()
    #
    # # use the same normal for all the points
    # # plane_normal = np.array([0.599, 0.800, -0.00436631])
    # # # use the same d for all the points
    # # d = 4.7
    # # # mirror the point use Householder transformation
    # # transform = np.eye(3) - (2 * plane_normal[:, None] * plane_normal[None, :])
    # # affine = transform @ reflect_points[:, :, None]
    # # affine += -2 * plane_normal[:, None] * d
    # # mirror_point = affine.squeeze()
    #
    # mirror_point_cloud = o3d.geometry.PointCloud()
    # mirror_point_cloud.points = o3d.utility.Vector3dVector(mirror_point)
    # o3d.visualization.draw_geometries([cloud_point_on_plane,reflect_points_cloud,mirror_point_cloud,mesh])

    #
    # # use scene.compute_distance to calculate the distance between the mirror point and the mesh
    # distance_mirror = scene.compute_distance(mirror_point.astype(np.float32))
    #
    # # if the distance between the mirror point and the mesh is less than 0.07, then the point is a reflection point
    # mask_hasnearmirror = distance_mirror < 0.07
    # indices = np.where(mask_isgreen & mask_outsidemesh)[0]
    # # update the label array using those indices
    # label[np.ravel(indices[mask_hasnearmirror])] = 2
    # label[np.ravel(indices[~mask_hasnearmirror])] = 3

    covariances[:, 2, 1] = label
    originpcd.covariances = o3d.utility.Matrix3dVector(covariances)
    return originpcd

start = 0
end = 99999
argv = sys.argv
start = int(argv[1])
end = int(argv[2])
print(start,end)

mesh = o3d.io.read_triangle_mesh("/mesh/labeledfinal.obj", True)
mesh_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
meshout = o3d.io.read_triangle_mesh("/mesh/meshout.ply")
meshout_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(meshout)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_for_raycast)
scene.add_triangles(meshout_for_raycast)
# visualize the mesh
# o3d.visualization.draw_geometries([mesh])

ouster_pose = np.loadtxt("/ouster_pose.txt")
hesai_pose = np.loadtxt("/hesai_pose.txt")
livox_pose = np.loadtxt("/livox_pose.txt")

path = "/processed/"
count = 0
#create raycast folder
if not os.path.exists(path+"raycast"):
    os.makedirs(path+"raycast")
#create each sensor folder
if not os.path.exists(path+"raycast/hesai"):
    os.makedirs(path+"raycast/hesai")
if not os.path.exists(path+"raycast/livox"):
    os.makedirs(path+"raycast/livox")
if not os.path.exists(path+"raycast/ouster"):
    os.makedirs(path+"raycast/ouster")

last_transformation = np.eye(4)


for i in range(max(start,0),min(end,len(ouster_pose))):#range(len(ouster_pose)):
    #every 10 choose one
    # if i<600:
    #     continue
    print(i)
    timestamp = ouster_pose[i, 0]
    trans_vec = ouster_pose[i, 1:4]
    print(trans_vec)

    # R = Rotation.as_matrix(Rotation.from_quat(pose[i, 4:]))
    closest_file_ouster = find_closest_ply(timestamp, path+"ouster")
    closest_file_hesai = find_closest_ply(timestamp, path+"hesai")
    closest_file_livox = find_closest_ply(timestamp, path+"livox")
    ouster_result_name = closest_file_ouster.split("/")[-1]
    hesai_result_name = closest_file_hesai.split("/")[-1]
    livox_result_name = closest_file_livox.split("/")[-1]
    # check if raycast file already exists
    # if os.path.exists(path+"raycast/ouster/"+ouster_result_name) and os.path.exists(path+"raycast/hesai/"+hesai_result_name) and os.path.exists(path+"raycast/livox/"+livox_result_name):
    #     print("already exists",i)
    #     continue

    pcd_ouster = read_point_cloud(closest_file_ouster)
    pcd_hesai = read_point_cloud(closest_file_hesai)
    pcd_livox = read_point_cloud(closest_file_livox)

    # ext_mat = extrinsic_matrix(R, trans_vec)
    # pcd_ouster = pcd_ouster.crop(o3d.geometry.AxisAlignedBoundingBox([-20, -20, -20], [20, 20, 20]))
    # pcd_hesai = pcd_hesai.crop(o3d.geometry.AxisAlignedBoundingBox([-20, -20, -20], [20, 20, 20]))
    # pcd_livox = pcd_livox.crop(o3d.geometry.AxisAlignedBoundingBox([-20, -20, -20], [20, 20, 20]))
    # write_point_cloud(path+"debug2.ply", pcd_ouster)

    # save covariances use copy
    # ouster_cov = copy.deepcopy(pcd_ouster.covariances)
    # hesai_cov = copy.deepcopy(pcd_hesai.covariances)
    # livox_cov = copy.deepcopy(pcd_livox.covariances)

    ouster_ext_mat = extrinsic_matrix(ouster_pose[i, 4:8], ouster_pose[i, 1:4])
    hesai_ext_mat = extrinsic_matrix(hesai_pose[i, 4:8], hesai_pose[i, 1:4])
    livox_ext_mat = extrinsic_matrix(livox_pose[i, 4:8], livox_pose[i, 1:4])

    # # calculate the relative transformation angle
    # relative_transformation = np.linalg.inv(last_transformation).dot(ext_mat)
    # # to euler angle in degree
    # relative_transformation_euler = Rotation.from_matrix(relative_transformation[:3,:3]).as_euler('xyz', degrees=True)
    # # print(relative_transformation_euler)
    # # if the abs of xyz angle is more than 5 degree, skip this frame
    # if np.any(np.abs(relative_transformation_euler)>2): #about 20% drop
    #     print("the relative transformation angle is too large",relative_transformation_euler,count)
    #     count+=1
    #     last_transformation = ext_mat
    #     continue
    # last_transformation = ext_mat

    # pcd_ouster.transform(ouster_ext_mat)
    # transform the hesai pointcloud with the extrinsic matrix and calibration matrix
    # hesai_ext_mat = ext_mat.dot(hesai_calib_mat)
    # pcd_hesai.transform(hesai_ext_mat)
    # transform the livox pointcloud with the extrinsic matrix and calibration matrix
    # livox_ext_mat = ext_mat.dot(livox_calib_mat)
    # pcd_livox.transform(livox_ext_mat)

    # readback the covariances
    # pcd_ouster.covariances = ouster_cov
    # pcd_hesai.covariances = hesai_cov
    # pcd_livox.covariances = livox_cov

    # write_point_cloud(path+"debug3.ply", pcd_ouster)

    # raycast_point for each lidar and save the result to separate folder
    ouster_result = raycast_point(pcd_ouster, ouster_ext_mat)
    hesai_result = raycast_point(pcd_hesai, hesai_ext_mat)
    livox_result = raycast_point(pcd_livox, livox_ext_mat)
    # save the raycast result from closest_file_ouster to another folder with same name

    write_point_cloud(path+"raycast/ouster/"+ouster_result_name, ouster_result)
    write_point_cloud(path+"raycast/hesai/"+hesai_result_name, hesai_result)
    write_point_cloud(path+"raycast/livox/"+livox_result_name, livox_result)
