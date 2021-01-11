# Humanoid Robot Imitation of Human Motion from Instructional Videos

## 1. Setup
* clone the repository
* install requirements

```bash
pip install -r requirements.txt
```

## 2. Train a policy

### a) Use VIBE to get poses 
TODO

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