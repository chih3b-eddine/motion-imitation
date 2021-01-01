import numpy as np
import quaternion
from math import radians
from mathutils import Matrix, Vector, Quaternion

bone_name_from_index = {
    0 : 'root',  # global body rotation
    9 : 'Spine2',     # chest
    1 : 'L_Hip',
    2 : 'R_Hip',
    3 : 'Spine1',     # abdomen
    4 : 'L_Knee',
    5 : 'R_Knee',
    7 : 'L_Ankle',
    8 : 'R_Ankle',
    13: 'L_Collar',   
    14: 'R_Collar',   
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
}

bone_names_ordered = ['root', "b'abdomen", 
                      "b'right_hip", "b'right_knee'", "b'right_ankle",
                      "b'left_hip",  "b'left_knee'",  "b'left_ankle",
                      "b'right_shoulder", "b'right_elbow'",
                      "b'left_shoulder", "b'left_elbow'"]
bone_index_ordered = [0, 3, 
                      2, 5, 8,
                      1, 4, 7,
                      17, 19,
                      16, 18]

positions_name_ordered = ['root', 'l_hand', 'r_hand', 'l_foot', 'r_foot']
positions_ids_ordered = [0,   # pelvis
                         36,  # lwrist
                         31,  # rwrist
                         21,  # OP LHeel    30, # lankle
                         24]  # OP RHeel    25  # rankle


def convert_position(pos):
    """
        switch axis (x, y, z) -> (y, z, x)
    """
    x, y, z = pos
    return np.asarray([y, z, x])


def Rodrigues(rotvec):
    """
            Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
            Source: smpl/plugins/blender/corrective_bpy_sh.py
    """
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

  
def aa_to_angle(aa):
    """
            Computes the angle of a rotation vector
    """
    return np.linalg.norm(aa)

def q_to_aa(q):
    """
            Computes rotation vector from a quaternion
    """
    rotvec = q.to_axis_angle() # get back to axis angle
    w_x, w_y, w_z, theta = rotvec[0][:], rotvec[1]
    return w__x*theta, w_y*theta, w_z*theta


rotVecDict = {
            "hip": [0, 0, 0],
            "hip": [0, 0, 0],
            "chest": [0, 1, 0],
            "neck": [0, 1, 0],
            "right hip": [0, -1, 0],
            "right knee": [0, 0, 0],
            "right ankle": [0, 0, 0],
            "right shoulder": [0, -1, 0],
            "right elbow": [0, 0, 0],
            "left hip": [0, -1, 0],
            "left knee": [0, 0, 0],
            "left ankle": [0, 0, 0],
            "left shoulder": [0, -1, 0],
            "left elbow": [0, 0, 0]
}

def compute_positions(joints3d, trans, offset, pelvis_position):
    joints3d = joints3d.reshape(-1,3)
    trans = convert_position(trans)
    positions = []
    
    positions.append(list(pelvis_position + trans - offset))
    for index in positions_ids_ordered[1:]:
        pos = convert_position(joints3d[index]) 
        positions.append(list(pos))
    return positions


def process_pose(pose):
    pose = pose.reshape(-1,3)
    angles = []
    
    quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-180))
    quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))
    quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))

    
    for index in bone_index_ordered:
        aa = pose[index,:]
        
        if index == 0 :              # root  z, x, y --> x, y, z
            q = Matrix(Rodrigues(aa)).to_quaternion()
            #q = quat_x_90_cw @ q
            #q = quat_y_90_cw @ q
            #q = (quat_x_90_cw @ quat_z_90_cw) @ q
            root_orientation = [q.x, -q.y, -q.z, q.w]
        
        elif index == 3:             # abdomen  z, y, x --> x, y, -z  ( +90 rotation around y) /// 1, 0, 2 --> 2 0 -1
            angles.append([aa[1]])
            angles.append([aa[0]])
            angles.append([aa[2]])

        elif index == 2:             # r-hip  x, z, y --> -y, z, x ( -90 rotation around z)  /// 2 1 0 --> -0 1 2
            angles.append([aa[2]])
            angles.append([aa[1]])
            angles.append([aa[0]])
            
        elif index == 1:             # l-hip  x, z, y --> y, z, -x ( +90 rotation around z)  /// 2 1 0 --> 0 1 -2
            angles.append([aa[2]])   
            angles.append([aa[1]])
            angles.append([aa[0]])
        
            
        elif index == 17:            # r-shoulder  x, y --> x, z    ( +90 rotation around x)  /// 2 0 --> 2 0
            angles.append([aa[2]])
            angles.append([aa[0]])    
            
        elif index == 16:            # l-shoulder  x, y --> x, -z  ( -90 rotation around x)  /// 2 0 --> 2 -0
            angles.append([aa[2]])
            angles.append([aa[0]])
            
        elif index in [8, 7]:        # ankles
            angles.append([aa[0]])
            angles.append([aa[2]])
            
        elif index in [5, 4]:        # knees 
            theta = aa_to_angle(aa)
            angles.append([-theta]) 
            
        elif index in [19, 18]:       # elbows
            theta = aa_to_angle(aa)
            angles.append([+theta])              
        
    return root_orientation, angles