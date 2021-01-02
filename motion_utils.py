import numpy as np
from math import radians
from mathutils import Matrix, Vector, Quaternion


# To compute Orientations from "VIBE-SMPL Poses"
bone_names_ordered = ['root', "b'abdomen", 
                      "b'right_hip", "b'right_knee'", "b'right_ankle",
                      "b'left_hip",  "b'left_knee'",  "b'left_ankle",
                      "b'right_shoulder", "b'right_elbow'",
                      "b'left_shoulder", "b'left_elbow'"]
bone_index_ordered = [9, 3, 
                      2, 5, 8,
                      1, 4, 7,
                      17, 19,
                      16, 18]


# To compute Positions from "VIBE-SMPL Joints3d"
positions_name_ordered = ['root', 'l_hand', 'r_hand', 'l_foot', 'r_foot']
positions_ids_ordered = [41,  # spine (H36M)
                         36,  # lwrist
                         31,  # rwrist
                         21,  # OP LHeel    30, # lankle
                         24]  # OP RHeel    25  # rankle



# Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues, Source: smpl/plugins/blender/corrective_bpy_sh.py
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)



# Computes the angle of a rotation vector (axis-angle representation)
def aa_to_angle(aa):
    return np.linalg.norm(aa)



# switch axis (x, y, z) -> (y, z, x)
def convert_position(pos):
    x, y, z = pos
    return np.asarray([y, z, x])



def compute_positions(joints3d, pelvis_position):
    """
        joints3d: (n_frames, 49, 3) SMPL 3D joints position in real world    
    returns:
        .. # TODO  / I can't 
    """
    positions = []
    
    # set absolute root position to BULLET PELVIS POSITION
    root_position = convert_position(joints3d[0])
    positions.append(list(root_position - pelvis_position))   
    
    # position of hand and feet
    for index in positions_ids_ordered[1:]:                     
        joint_position = convert_position(joints3d[index]) 
        positions.append(list(joint_position))
    return positions



def compute_orientations(pose):
    """
        pose: (n_frames, 72) array in axis-angle format [theta*wx,  theta*wy, theta*wz]
            pose[:3] : global body rotation (root=pelvis)
            pose[3:] : relative rotation of 23 joints
    returns:
        mapping of VIBE-SMPL body pose to Bullet orientations:
            - the global orientation is mapped from VIBE coordinate system (XYZ) to Bullet coordinate system (YZX), both expressed as
              (right,up,out) in right hand system
            - the joints orientation is already expressed relatively to the root, so no need to transform them        
    """
    pose = pose.reshape(-1,3)
    
    quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), radians(-90))  #  -90° rotation around X 
    quat_z_90_cw = Quaternion((0.0, 0.0, 1.0), radians(-90))
    quat_y_90_cw = Quaternion((0.0, 1.0, 0.0), radians(-90))
    
    angles = []
    for index in bone_index_ordered:
        aa = pose[index,:]  # joint pose in axis-angles representation
        
        if index == 9 :              # root  
            q = Matrix(Rodrigues(aa)).to_quaternion() # quaternion order =[w, x, y, z]
            #q = (quat_x_90_cw @ quat_z_90_cw) @ q
            root_orientation = [q.x, q.y, q.z, q.w]   # bullet quaternion order = [x, y, z, w]
        
        elif index == 3:             # abdomen /// bullet (z, y, x) --> Vibe (y, x, z) = (1, 0, 2)
            angles.append([aa[1]])
            angles.append([aa[0]])
            angles.append([aa[2]])

        elif index == 2:             # r-hip /// bullet (x, z, y) --> Vibe (z, y, x) = (2, 1, 0) 
            angles.append([aa[2]])
            angles.append([aa[1]])
            angles.append([aa[0]])
            
        elif index == 1:             # l-hip /// bullet (x, z, y) --> Vibe (z, y, x) = (2, 1, 0) 
            angles.append([aa[2]])   
            angles.append([aa[1]])
            angles.append([aa[0]])
            
        elif index == 17:            # r-shoulder /// bullet (x, y) --> Vibe (z, x) = (2, 0) 
            angles.append([aa[2]])
            angles.append([aa[0]])    
            
        elif index == 16:            # l-shoulder /// bullet (x, y) --> Vibe (z, x) = (2, 0) 
            angles.append([aa[2]])
            angles.append([aa[0]])
            
        elif index in [8, 7]:        # ankles   /// bullet (y, x) --> Vibe (x, z) = (0, 2)
            angles.append([aa[0]])
            angles.append([aa[2]])
            
        elif index in [5, 4]:        # knees 
            theta = aa_to_angle(aa)
            angles.append([-theta]) 
            
        elif index in [19, 18]:       # elbows
            theta = aa_to_angle(aa)
            angles.append([+theta])              
        
    return root_orientation, angles