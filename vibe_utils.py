TARGET_SMPL_JOINT_MAP ={
    'root': 0,    
    'chest': 9,
    'neck': 12,
    'r_hip': 2,
    'r_knee': 5,
    'r_ankle': 8,
    'r_shoulder': 17,
    'r_elbow': 19,
    'l_hip': 1,
    'l_knee': 4,
    'l_ankle': 7,
    'l_shoulder': 16,
    'l_elbow': 18,    
}



def get_vibe_person_keys():
  return [
          'pred_cam',
          'orig_cam',
          'verts',
          'pose',
          'betas',
          'joints3d',
          'joints2d',
          'bboxes',
          'frame_ids'
  ]

SMPL_JOINTS = {
    'hips': 0,            # root (I guess)
    'leftUpLeg': 1,
    'rightUpLeg': 2,
    'spine': 3,
    'leftLeg': 4,
    'rightLeg': 5,
    'spine1': 6,
    'leftFoot': 7,
    'rightFoot': 8,
    'spine2': 9,
    'leftToeBase': 10,
    'rightToeBase': 11,
    'neck': 12,
    'leftShoulder': 13,
    'rightShoulder': 14,
    'head': 15,
    'leftArm': 16,
    'rightArm': 17,
    'leftForeArm': 18,     
    'rightForeArm': 19, 
    'leftHand': 20,        
    'rightHand': 21,       
    'leftHandIndex1': 22,  
    'rightHandIndex1': 23
}

# this function is borrowed from https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/kp_utils.py
def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

# this function is borrowed from https://github.com/mkocabas/VIBE/blob/master/lib/data_utils/kp_utils.py
def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]

