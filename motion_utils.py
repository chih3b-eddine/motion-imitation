import numpy as np
from collections import OrderedDict


BULLET_SMPL_MAP = OrderedDict({
    'root'              : 0, # hips   
    "b'abdomen"         : 3,
    "b'right_hip"       : 2, 
    "b'right_knee'"     : 5,
    "b'right_ankle"     : 8,
    "b'left_hip"        : 1,
    "b'left_knee'"      : 4,
    "b'left_ankle"      : 7,
    "b'right_shoulder1" : 14,
    "b'right_shoulder2" : 17,
    "b'right_elbow'"    : 19,
    "b'left_elbow'"     : 18,
    "b'left_shoulder1"  : 13,
    "b'left_shoulder2"  : 16

})

JOINT_3D_MAP = {
    "root"   : 39, # hip
    "l_hand" : 36, # lwrist  
    "r_hand" : 31, # rwrit
    "l_foot" : 30, # lankle
    "r_foot" : 25  # rankle 
}


def aa_to_angle(aa):
    """
        aa = [w.theta] axis-angle -> theta (in radians)
    """ 
    return np.linalg.norm(aa)


def aa_to_quat(aa):
    """
        aa = [w.theta] axis-angle -> (s, x, y, z) unit quaternion 
    """
    angle = aa_to_angle(aa)
    w = np.true_divide(aa,(angle + 1e-8))   # avoid 0 angle
    s = np.cos(angle*0.5)
    w *= np.sin(angle*0.5)
    return np.insert(w, 0, s)


def quat_mult(q1, q2):
    """
        quaternions q1, q2  -> q1 x q2
    """
    s1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    s2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    x =  x1 * s2 + y1 * z2 - z1 * y2 + s1 * x2;
    y = -x1 * z2 + y1 * s2 + z1 * x2 + s1 * y2;
    z =  x1 * y2 - y1 * x2 + z1 * s2 + s1 * z2;
    s = -x1 * x2 - y1 * y2 - z1 * z2 + s1 * s2;
    return [s, x, y, z]


def quat_conj(q):
    """
        quaternion q  = (s, x, y, z)  -> q* = (s, -x, -y, -z)
    """
    s, x, y, z = q
    return [s, -x, -y, -z]


def compute_rotations(pose):
    """
    """
    v = []
    a = []
    
    ry = [0.7071, 0, 0.7071, 0]    # Quaternion that represents 90 degrees around Y
    rz = [0.7071, 0, 0, 0.7071]    # Quaternion that represents 90 degrees around Z
    
    rz_ = quat_conj(rz)
    ry_ = quat_conj(ry)
    
    for j, j_key in  BULLET_SMPL_MAP.items():
        aa = np.asarray(pose[j_key,:])
        if (j == 'root'):               # the bone towards -y / down
            q = aa_to_quat(aa)
            q = quat_mult(ry, q)              
            q = quat_mult(q, ry_)  
            q = quat_mult([0, 1, 0, 0], q)
            root_orientation = q
        
        elif (j == "b'abdomen"):        # the bone towards -z 
            q = aa_to_quat(aa)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)  
            a.extend([z, y, x])
            v.append(s)
            
        elif (j == "b'left_hip"):       # the bone towards +x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz)
            q = quat_mult(ry, q)
            s, x, y, z  = quat_mult(q, ry_)
            a.extend([x, z, y])
            v.append(s)
            
        elif (j == "b'right_hip"):      # the bone towards -x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz_)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([x, z, y])
            v.append(s)

        elif (j == "b'left_shoulder1"): # the bone towards +x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([x])
            v.append(s)
            
        elif (j == "b'left_shoulder2"): # the bone towards +x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([x])
            v.append(s)
            
        elif (j == "b'right_shoulder1"): # the bone towards -x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz_)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([x])
            v.append(s)
            
        elif (j == "b'right_shoulder2"): # the bone towards -x 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz_)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([x])
            v.append(s)

        elif (j == "b'left_ankle"):      # the bone towards +x  (error)
            q = aa_to_quat(aa)
            q = quat_mult(q, rz)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry)
            a.extend([y, x])
            v.append(s)
            #print(x, y, z)
            
        elif (j == "b'right_ankle"):       # the bone towards -x (error) 
            q = aa_to_quat(aa)
            q = quat_mult(q, rz_)
            q = quat_mult(ry, q)
            s, x, y, z = quat_mult(q, ry_)
            a.extend([y, x])
            v.append(s)
       
        elif (j in ["b'right_elbow'", "b'left_elbow'"]):
            s = aa_to_angle(aa)
            a.extend([s])
            v.append(s)
            
        elif (j in ["b'right_knee'", "b'left_knee'"]):
            s = aa_to_angle(aa)
            a.extend([-s])
            v.append(-s)
        
    return root_orientation, a, v
              

              