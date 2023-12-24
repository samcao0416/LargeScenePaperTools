import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys
from tqdm import tqdm

def vec2mat(q, trans_vec):
    ext_mat = np.eye(4)
    R = Rotation.as_matrix(Rotation.from_quat(q))
    ext_mat[:3, :3] = R
    ext_mat[:3, -1] = trans_vec
    return ext_mat

def mat2vec(ext_mat):
    q = Rotation.from_matrix(ext_mat[:3, :3]).as_quat()
    t = ext_mat[:3, -1]
    return q, t

if len(sys.argv) != 2:
    print("Usage: python vo_kf_trans.py <path>")
    exit(0)
    
# NOTE: copy the original vo_kf.txt and transform.txt to sys.argv[1] !!!!!!
insta_pose = np.loadtxt(os.path.join(sys.argv[1], "vo_kf.txt"))
# # save insta_pose
# np.savetxt(os.path.join(sys.argv[1], "result", "vo_kf.txt.bak"), insta_pose)
output_file = os.path.join(sys.argv[1], "vo_kf_trans.txt")
trans = np.loadtxt(os.path.join(sys.argv[1], "transform.txt"))

for i in tqdm(insta_pose):
    time = i[0]
    pose = vec2mat(i[4:8], i[1:4]) 
    
    # print(ext_mat)
    # transform using trans
    new_pose = np.matmul(trans, pose)
    q, t = mat2vec(new_pose)
    
    with open(output_file, "a") as f:
        f.write(str(time) + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " " + str(q[0]) + " " + str(q[1]) + " " + str(q[2]) + " " + str(q[3]) + "\n")

    