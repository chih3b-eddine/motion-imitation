import os
import joblib
import numpy as np
from collections import OrderedDict
from motion_utils import aa_to_angle, aa_to_quat, compute_rotations, BULLET_SMPL_MAP, JOINT_3D_MAP

FRAME_LENGTH = 1/30
input_file = "vibe_output.pkl"


# load VIBE tracking results
if not os.path.isfile(input_file):
        exit(f'vibe output file \"{input_file}\" does not exist!')

data = joblib.load(input_file)
if data is None:
    exit(f'cannot load\"{input_file}"!')

    
# convert SMPL to Bullet 
motion = []
t_prev = 1
s_prev = [0]*13
a_prev = [0]*21
for person_id, person_data in data.items():
    
    poses = person_data['pose']               # Nx72 = Nx24x3
    orig_cam = person_data['orig_cam']        # in original image space Nx4 : (sx,sy,tx,ty)
    joints3ds = person_data['joints3d']        # Nx24x3 SMPL 3D joints  
    frames = person_data['frame_ids']
   
    for i in range(len(poses)):
        t = frames[i]

        joints3d = joints3ds[i].reshape(-1,3)
        pose = poses[i].reshape(-1,3)
        
        # get positions
        root_pos   = joints3d[JOINT_3D_MAP["root"]]
        l_hand_pos = joints3d[JOINT_3D_MAP["l_hand"]]
        r_hand_pos = joints3d[JOINT_3D_MAP["r_hand"]]
        l_foot_pos = joints3d[JOINT_3D_MAP["l_foot"]]
        r_foot_pos = joints3d[JOINT_3D_MAP["r_foot"]]
        
        # get angles and velocities
        root_ori, joints_a, joints_s = compute_rotations(pose) 
        
        if (t==1):
            s_prev = joints_s 
            a_prev = joints_a
            
        joints_angular_v = np.subtract(joints_s, s_prev)  # joints angular velocities  #13
        joints_v = np.subtract(joints_a, a_prev)          # joints linear velocities   #21
        
        s_prev = joints_s
        a_prev = joints_v
        
        output_dict = OrderedDict({
            "frame_id"          : (t-t_prev)*FRAME_LENGTH,  # seconds
            "jointsAngles"      : joints_a,     # (21,) in radians
            "jointsVelocities"  : joints_v,     # (21,)  finite difference 
            "rootPosition"      : root_pos,     # (x,y,z) real world 
            "rootOrientation"   : root_ori,     # (a,b,c,d) quat real world 
            "leftHandPosition"  : l_hand_pos,   # (x,y,z) real world 
            "rightHandPosition" : r_hand_pos,   # (x,y,z) real world
            "leftFootPosition"  : l_foot_pos,   # (x,y,z) real world
            "rightFootPosition" : r_hand_pos,   # (x,y,z) real world
})
        t_prev = t
        motion.append(output_dict)
  
    
# save motion
with open("bullet_motion.pkl", "wb") as f:
    joblib.dump(motion, f)

    
# ------------------------------------------
#import joblib
#data = joblib.load("bullet_motion.pkl")
#print(len(data))       : 5504
#for k, v in data[0].items():
#    if (k != "frame_id"):
#        print(k, len(v))  

