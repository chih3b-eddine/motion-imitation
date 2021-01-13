# Humanoid Robot Imitation of Human Motion from Instructional Videos

## 1. Setup
* clone the repository
* install requirements

```bash
pip install -r requirements.txt
```

## 2. Train a policy

### a) Use VIBE to get poses 
* in `run_vibe.ipynb` update the ID of the Youtube video from which you wish to extract the poses.

```python
YOUTUBE_ID = 'Ae3AkGYpWsM'
```
* Eventually select a specific part of the video. In the example below, extract "7" seconds  from "youtube.mp4" starting from the second "36" and save the result to "video.mp4" 

```python
!ffmpeg -y -loglevel info -i youtube.mp4 -ss 00:00:36 -t 7 video.mp4

```
* Update the output folder where VIBE results would be saved. 
```
!python demo.py --vid_file video.mp4 --output_folder /output
```
* In the output folder, VIBE  generates a  subfolder with the basename of the input video where it saves 2 files:
- "basename_vibe_result.mp4" : a video of the rendering of tracked poses
- "basename_vibe_output.pkl" : the tracking results in SMPL format

### b) Map these poses to Bullet environment
* in `motion.py` update the path to vibe output

```python
data_folder = "data"
input_file = "walking_vibe_output.pkl"
```

* run `motion.py` to generate a json file containing the poses with Bullet environment format

```bash
python motion.py
```

### c) Train the humanoid to preform the video motion
* in `learning.py` update the path to the json data output form the previous step

```python
path_to_data = "data/walking.json"
```

* run `learning.py`
```bash
python learning.py
```
