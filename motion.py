import os
import joblib
import json
import numpy as np
from motion_utils import  convert_position, process_pose


FPS_VIBE = 30  # frames per second in VIBE
BULLET_GRAVITY =  np.array([0,0,-9.81])

JOINT_3D_MAP = {
    "root"   : 39, # hip
    "l_hand" : 36, # lwrist  
    "r_hand" : 31, # rwrit
    "l_foot" : 30, # lankle
    "r_foot" : 25  # rankle 
}

def process_poses(data, person_id=1):
    person_data = data[person_id]

    frames = person_data['frame_ids']
    joints3ds = person_data['joints3d']       # Nx24x3 SMPL 3D joints  
    poses = person_data['pose']               # Nx72 = Nx24x3
    
    motion = []
    for i in range(len(poses)):        
        frame_id = frames[i]
        joints3d = joints3ds[i].reshape(-1,3)
        pose = poses[i].reshape(-1,3)
        
        # get positions
        root_pos   = convert_position(joints3d[JOINT_3D_MAP["root"]]) + BULLET_GRAVITY
        l_hand_pos = convert_position(joints3d[JOINT_3D_MAP["l_hand"]])
        r_hand_pos = convert_position(joints3d[JOINT_3D_MAP["r_hand"]])
        l_foot_pos = convert_position(joints3d[JOINT_3D_MAP["l_foot"]])
        r_foot_pos = convert_position(joints3d[JOINT_3D_MAP["r_foot"]])
        
        # get orientations
        root_orientation, joints_a = process_pose(pose)
        
        if (i==0):
            root_origin = root_orientation
        
        # compute root orientation
        root_orientation = np.subtract(root_orientation, root_origin) # + BULLET_GRAVITY orientation
        
        # compute velocities
        if (i==0):
            velocities = [[0] for k in joints_a]
        else:
            delta_t = (frame_id - frame_id_prev)/FPS_VIBE
            a_c = [k[0] for k in joints_a]
            velocities = np.subtract(a_c, a_prev)/delta_t
            velocities = [[k] for k in velocities]
        
        # keep track of current orientations for next pose velocities
        a_prev = [k[0] for k in joints_a]
        frame_id_prev = frame_id
        
        motion.append({
            "frame_id"          : frame_id,  
            "jointsAngles"      : joints_a,     
            "jointsVelocities"  : velocities,     
            "rootPosition"      : list(root_pos),     
            "rootOrientation"   : root_orientation,    
            "leftHandPosition"  : list(l_hand_pos),   
            "rightHandPosition" : list(r_hand_pos),   
            "leftFootPosition"  : list(l_foot_pos),   
            "rightFootPosition" : list(r_hand_pos),   
        })
    return {"timestep": 1/FPS_VIBE, "frames": motion}



if __name__ == "__main__":
    input_file = "vibe_output.pkl"
    if not os.path.isfile(input_file):
        exit(f'vibe output file \"{input_file}\" does not exist!')
    
    data = joblib.load(input_file)
    if data is None:
        exit(f'cannot load\"{input_file}"!')
        
    result = process_poses(data, 1)
    
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()
        
    with open('data.json', 'w') as f:
        json.dump(result, f, indent=4, default=np_encoder)
