import os
import joblib
import json
import numpy as np
from motion_utils import compute_orientations, compute_positions


data_folder = "data"
input_file = "cartwheel_vibe_output.pkl"

# VIBE parameters
FPS_VIBE = 30  # frames per second 

# BULLET parameters
PELVIS_POSITION = np.array([0, 0, -0.165])        
PELVIS_ORIENTATION = [1.000, 0, -0.002, 0]


def process_poses(data, person_id=1):
    print("Processing frames of person %d ..."%person_id)
    
    person_data = data[person_id]
    frames   = person_data['frame_ids']
    joints3ds = person_data['joints3d']       # (n_frames, 49, 3) SMPL 3D joints   
    poses = person_data['pose']               # (n_frames, 72) SMPL poses  72 = (1 root+ 23 joints) orientation parameters     
     
    motion = []
    for i in range(len(poses)):
        if (i%50==0): print(" %d frames processed"%i)
        frame = frames[i]
        
        # compute orientations
        root_orientation, angles = compute_orientations(poses[i])
        
        # compute positions 
        positions = compute_positions(joints3ds[i], PELVIS_POSITION)

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
    print(" %d Total frames processed"%i)
    
    result = {
        "timestep": 1/FPS_VIBE, 
        "frames": motion
    }
    if result is None:
        exit("Failed to process frames")
    return result


def save_data(result, input_file):
    output_path = os.path.join(data_folder, os.path.basename(input_file).replace('_vibe_output.pkl', '.json'))
    
    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4, default=np_encoder)
    print(f'Data saved succesfully to \"{output_path}\"')

    
def load_data(input_file):
    input_path = os.path.join(data_folder, input_file)
    if not os.path.isfile(input_path):
        exit(f'vibe output file \"{input_path}\" does not exist!') 
        
    data = joblib.load(input_path)
    if data is None:
        exit(f'cannot load\"{input_path}"!')
    print(f'Data loaded successfully from \"{input_path}\"')  
    return data


if __name__ == "__main__":
    # load data
    data = load_data(input_file)
    
    # process data
    result = process_poses(data, 1)
    
    # save data
    save_data(result, input_file)
   