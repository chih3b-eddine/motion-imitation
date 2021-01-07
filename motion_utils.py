import numpy as np
from math import radians
from mathutils import Matrix, Vector, Quaternion
import mathutils as mt


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
positions_ids_ordered = [40,  # thorax
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


def quat_to_aa(q):
    return q.w*q.x, q.w*q.y, q.w*q.z


def quat_invert(q):
    return Quaternion((q.w, -q.x, -q.y, -q.z))


# switch axis (x, y, z) -> (y, z, x)
def convert_position(pos):
    x, y, z = pos
    return np.asarray([y, z, x])


def compute_positions(joints3d, root0, chest0):
    """
        joints3d: (n_frames, 49, 3) SMPL 3D joints position in real world    
    returns:
        absolute position of root, hand and feet along time by:
            - mapping the SMPL-root to BULLET-chest (the chest is the origin of BULLET space) at the first timestep
            - computing the position of each joint shifted by the translation of the mapped SMPL-root 
    """
    positions = []
    
    shift = np.asarray([0, 0, 0])
    root0 = convert_position(root0)
    chest0 = convert_position(chest0)
    root_position = convert_position(joints3d[0])
   
    for index in positions_ids_ordered:                     
        joint_position = convert_position(joints3d[index]) 
        positions.append(list(joint_position + root_position - root0 - chest0 + shift))   
    return positions



def compute_orientations(pose):
    """
        pose: (n_frames, 72) array in axis-angle format [theta*wx,  theta*wy, theta*wz]
            pose[:3] : global body rotation (root=pelvis)
            pose[3:] : relative rotation of 23 joints
    returns:
        mapping of VIBE-SMPL body pose to Bullet orientations:
            - change the origin of space from VIBE-root to BULLET-chest
            - change the global orientation from VIBE coordinate system (YZX) to Bullet coordinate system (XYZ), both expressed as
              (out, right, up) in right hand system
            - the joints orientation is already expressed relatively to their parents, so no need to transform them        
    """
    pose = pose.reshape(-1,3)
    
    rotation_x_ = Quaternion((1.0, 0.0, 0.0), radians(-90))
    rotation_y_ = Quaternion((0.0, 1.0, 0.0), radians(-90))
    rotation_z_ = Quaternion((0.0, 0.0, 1.0), radians(-90))

    
    G_root = Matrix(Rodrigues( pose[0,:])).to_quaternion()      # root global orientation
    q_spine = Matrix(Rodrigues( pose[3,:])).to_quaternion()
    q_spine_1 = Matrix(Rodrigues( pose[6,:])).to_quaternion()
    q_spine_2 = Matrix(Rodrigues( pose[9,:])).to_quaternion()
    G_chest = G_root @ (q_spine @ (q_spine_1 @ q_spine_2))      # chest global orientation
    
    G_chest_inv = quat_invert(G_chest)
    
    global_rotation = ( quat_invert(rotation_z_) @ rotation_z_)
    
    angles = []
    for index in bone_index_ordered:
        aa = pose[index,:]           # joint pose in axis-angles representation
        q = Matrix(Rodrigues(aa)).to_quaternion()
        
        if index == 9 :              # chest 
            q = global_rotation @ G_chest                    # mathutils quaternion order = [w, x, y, z]
            root_orientation = [q.x, q.y, q.z, q.w]          # bullet quaternion order = [x, y, z, w]
            
        elif index == 3:             # abdomen /// bullet (z, y, x) 
            q = G_chest_inv @ (G_root @ q)
            q = global_rotation @ q
            aa = quat_to_aa(q)
            
            angles.append([aa[2]])
            angles.append([aa[1]])
            angles.append([aa[0]])

        elif index == 2:             # r-hip /// bullet (x, z, y)
            q = G_chest_inv @ (G_root @ q)
            q = global_rotation @ q
            aa = quat_to_aa(q)
            
            angles.append([aa[0]])
            angles.append([aa[2]])
            angles.append([aa[1]])
            
        elif index == 1:             # l-hip /// bullet (x, z, y) 
            q = G_chest_inv @ (G_root @ q)
            q = global_rotation @ q
            aa = quat_to_aa(q)
            
            angles.append([aa[0]])   
            angles.append([aa[2]])
            angles.append([aa[1]])
            
        elif index == 17:            # r-shoulder /// bullet (x, y) 
            q_parent = Matrix(Rodrigues( pose[14,:])).to_quaternion()
            q = G_chest_inv @ (q_spine_2 @ (q_parent @ q))
            q = global_rotation @ q
            aa = quat_to_aa(q)
            
            angles.append([aa[0]])
            angles.append([aa[1]])    
            
        elif index == 16:            # l-shoulder /// bullet (x, y) 
            q_parent = Matrix(Rodrigues( pose[13,:])).to_quaternion()
            q = G_chest_inv @ (q_spine_2 @ (q_parent @ q))
            q = global_rotation @ q
            aa = quat_to_aa(q)
            
            angles.append([aa[0]])
            angles.append([aa[1]])
            
        elif index in [8, 7]:        # ankles   /// bullet (y, x) 
            q = global_rotation @ q
            aa = quat_to_aa(q)
            angles.append([aa[1]])
            angles.append([aa[0]])
            
        elif index in [5, 4]:        # knees 
            theta = aa_to_angle(aa)
            angles.append([-theta]) 
            
        elif index in [19, 18]:       # elbows
            theta = aa_to_angle(aa)
            angles.append([theta])              
        
    return root_orientation, angles