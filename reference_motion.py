from env import RLEnv
import json
import time
       
if __name__ == "__main__":
    path_to_data = "data/dance.json"
    
    with open(path_to_data, "r") as f:
        reference_motion = json.loads(f.read())
    
    frames = reference_motion["frames"]
    n_frames = len(frames)
    
    env = RLEnv(initial_state=frames[0])
    for t in range(1, n_frames):
            state = frames[t]
            env.test_passive(state)
            time.sleep(0.5)