import open3d as o3d
import numpy as np
from scipy.spatial.transform.rotation import Rotation
import time
import matplotlib.pyplot as plt
import cv2
import glob
import os
def extrinsic_matrix(rot_mat, trans_vec):
    ext_mat = np.eye(4)
    ext_mat[:3, :3] = rot_mat
    ext_mat[:3, -1] = trans_vec
    return ext_mat

def intrinsic_matrix(fov, width, height):
    int_mat = np.zeros((3, 3))
    f = width / (2 * np.tan(fov / 2))
    int_mat[0, 0] = f
    int_mat[1, 1] = f
    int_mat[2, 2] = 1
    int_mat[0, 2] = width / 2
    int_mat[1, 2] = height / 2
    return int_mat

def find_closest_image(timestamp,path):
    pcd_files = sorted(glob.glob(path + "/*.jpg"))
    min_diff = float("inf")
    closest_file = None
    for pcd_file in pcd_files:
        # file_timestamp = float(os.path.basename(pcd_file).split(".")[0])/1000000
        # if "ouster" in pcd_file:
        file_timestamp = float(os.path.basename(pcd_file).split(".")[0])+float(os.path.basename(pcd_file).split(".")[1])/1000000
        diff = abs(file_timestamp - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_file = pcd_file
    print(closest_file, timestamp)
    if min_diff>0.1:
        print("the min_diff is too large",min_diff,timestamp,closest_file)
        closest_file = None
    return closest_file

fov = np.radians(90)
width = 1000
height = 1000
int_mat = intrinsic_matrix(fov, width, height)

mesh = o3d.io.read_triangle_mesh("mesh/labeledfinal.obj", True)
vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height)
vis.add_geometry(mesh)
# vis.get_render_option().load_from_json("../../test_data/renderoption.json")
imagepath = "images"
pose = np.loadtxt("vo_kf.txt")
#read calibration file, opencv fisheye calib in format of [ h w fx fy cx cy k1 k2 p1 p2 k3]
cv_file = cv2.FileStorage("/home/zxt/public/zxt/reflectiondatasetfinal/camera.yaml", cv2.FILE_STORAGE_READ)
image_width = cv_file.getNode('width').real()
image_height = cv_file.getNode('height').real()
camera_intrinsic = cv_file.getNode('intrinsic').mat()
camera_distortion = cv_file.getNode('distortion').mat()

ctr = vis.get_view_control()


for i in range(len(pose)):
    print(i)
    trans_vec = pose[i, 1:4]
    R = Rotation.as_matrix(Rotation.from_quat(pose[i, 4:]))
    timestamp = pose[i, 0]

    ext_mat = extrinsic_matrix(R, trans_vec)
    inv_ext = np.linalg.inv(ext_mat)
    # print(inv_ext)

    pinhole_camera = o3d.camera.PinholeCameraParameters()
    pinhole_camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, int_mat[0, 0], int_mat[1, 1],
                                                             int_mat[0, 2], int_mat[1, 2])
    pinhole_camera.extrinsic = inv_ext
    ctr.convert_from_pinhole_camera_parameters(pinhole_camera, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    rgb = np.asarray(vis.capture_screen_float_buffer(True))
    #get three mask image from rgb, [0,255,0] is glass, [0,0,255] is mirror, [0,255,255] is othersref, each is image with 0 and 1
    mask_glass = np.zeros((height, width))
    mask_mirror = np.zeros((height, width))
    mask_others = np.zeros((height, width))
    mask_glass[np.where((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 1) & (rgb[:, :, 2] == 0))] = 1
    mask_mirror[np.where((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 1))] = 1
    mask_others[np.where((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 1) & (rgb[:, :, 2] == 1))] = 1

    #visulize the mask in one image with different color
    mask = np.zeros((height, width, 3))
    # #filter the mask
    # mask_glass = cv2.medianBlur(mask_glass.astype(np.uint8), 5)
    # mask_mirror = cv2.medianBlur(mask_mirror.astype(np.uint8), 5)
    # mask_others = cv2.medianBlur(mask_others.astype(np.uint8), 5)



    #filter the mask with erode and dilate
    kernel = np.ones((6, 6),np.uint8)
    mask_glass = cv2.dilate(mask_glass.astype(np.uint8),kernel,iterations = 1)
    mask_glass = cv2.erode(mask_glass.astype(np.uint8),kernel,iterations = 1)
    mask_mirror = cv2.dilate(mask_mirror.astype(np.uint8),kernel,iterations = 1)
    mask_mirror = cv2.erode(mask_mirror.astype(np.uint8),kernel,iterations = 1)
    mask_others = cv2.dilate(mask_others.astype(np.uint8),kernel,iterations = 1)
    mask_others = cv2.erode(mask_others.astype(np.uint8),kernel,iterations = 1)
    # detect connected component and filter the small connected component
    # print(np.sum(mask_glass),np.sum(mask_mirror),np.sum(mask_others))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_glass.astype(np.uint8), connectivity=8)
    # remove the small area with pixel number less than 50
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            mask_glass[labels == i] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_mirror.astype(np.uint8), connectivity=8)
    # print(num_labels, stats, centroids, np.sum(mask_mirror),timestamp,labels)
    # remove the small area with pixel number less than 50
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            mask_mirror[labels == i] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_others.astype(np.uint8), connectivity=8)
    # remove the small area with pixel number less than 50
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            mask_others[labels == i] = 0
    # print(np.sum(mask_glass),np.sum(mask_mirror),np.sum(mask_others))

    # mask[:, :, 0] = mask_glass
    # mask[:, :, 1] = mask_mirror
    # mask[:, :, 2] = mask_others
    print(timestamp, np.sum(mask_mirror))


    mask_all = mask_glass + mask_mirror + mask_others
    mask_all[mask_all > 1] = 1
    #show image use plt
    # plt.imshow(mask)
    # plt.show()
    #get the corresponding image
    imagefile = find_closest_image(timestamp,imagepath)
    if imagefile is None:
        continue
    image = cv2.imread(imagefile)
    # undistort image
    # Knew is open3d intrinsic matrix, convert to opencv intrinsic matrix
    Knew = pinhole_camera.intrinsic.intrinsic_matrix
    image = cv2.fisheye.undistortImage(image, K=camera_intrinsic, D=camera_distortion, Knew=Knew, new_size=(int(width), int(height)))
    #show image use plt
    # plt.imshow(image)
    # plt.show()
    #make save folder
    if not os.path.exists("rgb/alllabel"):
        os.makedirs("rgb/alllabel")
    # if not os.path.exists("rgb/glass"):
    #     os.makedirs("rgb/glass")
    # if not os.path.exists("rgb/mirror"):
    #     os.makedirs("rgb/mirror")
    # if not os.path.exists("rgb/others"):
    #     os.makedirs("rgb/others")
    #make image and mask subfolder in each folder
    if not os.path.exists("rgb/alllabel/image"):
        os.makedirs("rgb/alllabel/image")
    if not os.path.exists("rgb/alllabel/mask"):
        os.makedirs("rgb/alllabel/mask")
    # if not os.path.exists("rgb/glass/image"):
    #     os.makedirs("rgb/glass/image")
    # if not os.path.exists("rgb/glass/mask"):
    #     os.makedirs("rgb/glass/mask")
    # if not os.path.exists("rgb/mirror/image"):
    #     os.makedirs("rgb/mirror/image")
    # if not os.path.exists("rgb/mirror/mask"):
    #     os.makedirs("rgb/mirror/mask")
    # if not os.path.exists("rgb/others/image"):
    #     os.makedirs("rgb/others/image")
    # if not os.path.exists("rgb/others/mask"):
    #     os.makedirs("rgb/others/mask")
    # if mask_glass not zero, save image and mask to glass folder
    if np.sum(mask_all) > 200:
        cv2.imwrite("rgb/alllabel/image/"+os.path.basename(imagefile).replace(".jpg",".png"),image)
        cv2.imwrite("rgb/alllabel/mask/"+os.path.basename(imagefile).replace(".jpg",".png"),mask_all*255)
    # if np.sum(mask_glass) > 200:
    #     cv2.imwrite("rgb/glass/image/" + os.path.basename(imagefile).replace(".jpg",".png"), image)
    #     cv2.imwrite("rgb/glass/mask/" + os.path.basename(imagefile).replace(".jpg",".png"),mask_glass * 255)
    # if np.sum(mask_mirror) > 200:
    #     cv2.imwrite("rgb/mirror/image/" + os.path.basename(imagefile).replace(".jpg",".png"), image)
    #     cv2.imwrite("rgb/mirror/mask/" + os.path.basename(imagefile).replace(".jpg",".png"),mask_mirror * 255)
    # if np.sum(mask_others) > 200:
    #     cv2.imwrite("rgb/others/image/" + os.path.basename(imagefile).replace(".jpg",".png"), image)
    #     cv2.imwrite("rgb/others/mask/" + os.path.basename(imagefile).replace(".jpg",".png"),mask_others * 255)

    #save three mask to three different folder use the same name as image png file, and save corresponding undisort image to the undistort folder
    # cv2.imwrite("rgb/undistort/"+os.path.basename(imagefile),image)
    # cv2.imwrite("rgb/glass/"+os.path.basename(imagefile).replace(".jpg",".png"),mask_glass*255)
    # cv2.imwrite("rgb/mirror/"+os.path.basename(imagefile).replace(".jpg",".png"),mask_mirror*255)
    # cv2.imwrite("rgb/others/"+os.path.basename(imagefile).replace(".jpg",".png"),mask_others*255)



    # time.sleep(1 / 2)
# vis.run()
vis.destroy_window()
