import numpy as np
import quaternion
from math import radians
from mathutils import Matrix, Vector, Quaternion

bone_name_from_index = {
    0 : 'root',  
    1 : 'L_Hip',
    2 : 'R_Hip',
    3 : 'Spine1',     # abdomen
    4 : 'L_Knee',
    5 : 'R_Knee',
    7 : 'L_Ankle',
    8 : 'R_Ankle',
    13: 'L_Collar',   # l_shoulder1 
    14: 'R_Collar',   # r_shoulder1
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
}

bone_names_ordered = ['root', "b'abdomen", 
                      "b'right_hip", "b'right_knee'", "b'right_ankle",
                      "b'left_hip",  "b'left_knee'",  "b'left_ankle",
                      "b'right_shoulder1'", "b'right_shoulder2'", 
                      "b'right_elbow'",    "b'left_elbow'",
                      "b'left_shoulder1'",  "b'left_shoulder2'"]

bone_index_ordered = [0, 3, 
                      2, 5, 8,
                      1, 4, 7,
                      14, 17, 
                      19, 18, 
                      13, 16]


# switch axis [x, y, z] -> [y, z, x]
def convert_position(pos):
    return [pos[1], pos[2], pos[0]]


# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
# Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

  
# compute the angle of an angle-axis
def aa_to_angle(aa):
    return np.linalg.norm(aa)


def process_pose(pose):
    angles = []
    for index in bone_index_ordered:
        aa = pose[index,:]

        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))  # -90 degrees rotation around X
        quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))  # -90 degrees rotation around Y
        quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))  # -90 degrees rotation around Z
        quat_yn_90_cw = Quaternion((0.0, 1.0, 0.0), radians(90))  # +90 degrees rotation around Y
        
        if index == 0 :  
            # ---------- root : ZXY to XYZ --------------------#
            # -----------------------------------------
            # SMPL Root: (x, y, z) = (right, up, out)   
            # BULLET Root: (x, y, z) = (out, right, up)  
            # -----------------------------------------
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = (quat_z_90_cw @ quat_x_90_cw) @ bone_rotation
            root_orientation = [q.w, q.x, q.y, q.z]
        
            #---------------- TOP PARTS ------------------------#
        elif index == 3:
            # ---------- abdomen : rotation vector +Y ----------#
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = (quat_yn_90_cw) @ bone_rotation             
            angles.append([q.z])
            angles.append([q.y])
            angles.append([q.x])

        elif index == 1:
            # ---------- l-shoulder : rotation vector -Y -------#
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = quat_y_90_cw @ bone_rotation
            q = (quat_z_90_cw @ quat_x_90_cw) @ q
            angles.append([q.x])
            angles.append([q.z])
            angles.append([q.y])

        elif index == 2:
            # ---------- r-shoulder : rotation vector +Y------#
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = quat_yn_90_cw @ bone_rotation
            q = (quat_z_90_cw @ quat_x_90_cw) @ q
            angles.append([q.x])
            angles.append([q.z])
            angles.append([q.y])
            
            #------------------- DOWN PARTS ------------------#
        elif index == 1:
            # ---------- l-hip : rotation vector -Y ----------#
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = quat_y_90_cw @ bone_rotation
            q = (quat_z_90_cw @ quat_x_90_cw) @ q
            angles.append([q.x])
            angles.append([q.z])
            angles.append([q.y])

        elif index == 2:
            # ---------- r-hip : rotation vector +Y ----------#
            bone_rotation = Matrix(Rodrigues(aa)).to_quaternion()
            q = quat_yn_90_cw @ bone_rotation
            q = (quat_z_90_cw @ quat_x_90_cw) @ q
            angles.append([q.x])
            angles.append([q.z])
            angles.append([q.y])
            
        elif index in [4, 5]:
            # ---------- knees : rotation -Y -----------------#
            q = aa_to_angle(aa)
            angles.append([- q]) 
        
        elif index in [7, 8]:
            # ---------- ankles  ----------#
            q = Matrix(Rodrigues(aa)).to_quaternion()
            q = (quat_z_90_cw @ quat_x_90_cw) @ q
            angles.append([q.y])
            angles.append([q.x])
        else:
            q = aa_to_angle(aa)
            angles.append([q])                  
        
    return root_orientation, angles