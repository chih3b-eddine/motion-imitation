import os
import joblib
import json
import numpy as np
from motion_utils import  convert_position, process_pose, compute_positions


FPS_VIBE = 30  # frames per second in VIBE

# Target parameters
PELVIS_POSITION = np.array([0, 0, -0.165])        
PELVIS_ORIENTATION = [1.000, 0, -0.002, 0]


positions_name_ordered = ['root', 'l_hand', 'r_hand', 'l_foot', 'r_foot']
positions_ids_ordered = [0,   # pelvis
                         36,  # lwrist
                         31,  # rwrist
                         21,  # OP LHeel    30, # lankle
                         24]  # OP RHeel    25  # rankle


def process_poses(data, person_id=1):
    person_data = data[person_id]

    frames   = person_data['frame_ids']
    joints3ds = person_data['joints3d']       # Nx24x3 SMPL 3D joints  
    
    poses = person_data['pose']               # Nx72 : 72= 3 global orientation parameters + 23 joints orientation    
    
    trans = np.asarray([list(p[0]) for p in joints3ds])  # root positions along the frames
    offset = np.array([trans[0][0], trans[0][1], trans[0][1]])
    
    
    motion = []
    for i in range(len(poses)):
        print('Adding pose: ' + str(i))
        frame = frames[i]
        
        # compute orientations
        root_orientation, angles = process_pose(poses[i])
        
        # compute positions 
        positions = compute_positions(joints3ds[i], trans[i], offset, PELVIS_POSITION)

        # compute velocities
        if (i==0): 
            velocities = [[0] for v in angles]
        else:
            duration = (frame - previous_frame)/FPS_VIBE
            velocities = (np.hstack(angles) - np.hstack(previous_angles))/duration
            velocities = [[v] for v in velocities]
        
        previous_frame = frame
        previous_angles = angles
              
  
        motion.append({
            "frame_id"          : frame,  
            "jointsAngles"      : angles,     
            "jointsVelocities"  : velocities,     
            "rootPosition"      : positions[0],    
            "rootOrientation"   : root_orientation,    
            "leftHandPosition"  : positions[1],   
            "rightHandPosition" : positions[2],   
            "leftFootPosition"  : positions[3],   
            "rightFootPosition" : positions[4],   
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
