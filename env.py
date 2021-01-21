import pybullet
import time
import pybullet_data
import numpy as np
import json

class RLEnv:
    def __init__(self, initial_state=None, useGUI=True, timeStep=1/30):
        self.useGUI = useGUI
        if self.useGUI:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.timeStep = timeStep
        self.humanoid = None
        self.jointIds = None
        self.jointNames = None
        self.terminalLinksIds = [27, 32, 13, 22] ## right hand, left hand, right foot, left foot
        self.reset(initial_state)
        return

    def reset(self, initial_state=None):
        pybullet.resetSimulation()
        pybullet.setPhysicsEngineParameter(fixedTimeStep=self.timeStep, numSubSteps=2)
        pybullet.setGravity(0,0,-9.81)
        obUids = pybullet.loadMJCF("mjcf/humanoid.xml")
        self.humanoid = obUids[1]
        self.jointIds = []
        self.jointNames = []

        for j in range(pybullet.getNumJoints(self.humanoid)):
            pybullet.changeDynamics(self.humanoid, j, linearDamping=0, angularDamping=0)
            info = pybullet.getJointInfo(self.humanoid, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == pybullet.JOINT_SPHERICAL or jointType == pybullet.JOINT_REVOLUTE):
                self.jointIds.append(j)
                self.jointNames.append(jointName)

        if initial_state:
            pybullet.resetBasePositionAndOrientation(self.humanoid, initial_state["rootPosition"], initial_state["rootOrientation"])
            pybullet.resetJointStatesMultiDof(self.humanoid, self.jointIds, initial_state["jointsAngles"], initial_state["jointsVelocities"])
            
        return self.get_state()

    def dim_action(self):
        return len(self.jointIds)

    def dim_state(self):
        return 2*len(self.jointIds) + 3*5 + 2 + 4

    def normalize_action(self, action):
        s = action.sign()
        r = action.abs() - (action.abs()//(2*np.pi))*2*np.pi
        r -= (r>np.pi)*2*np.pi
        return r*s

    def step(self, action, reference_motion):
        action = self.normalize_action(action)
        pybullet.setJointMotorControlArray(self.humanoid, self.jointIds, pybullet.POSITION_CONTROL, action)
        pybullet.stepSimulation()
        next_state = self.get_state()
        reward = self.compute_reward(next_state, reference_motion)
        return next_state, reward

    def get_state(self):
        jointStates = pybullet.getJointStates(self.humanoid, self.jointIds)
        rootPosition, rootOrientation = pybullet.getBasePositionAndOrientation(self.humanoid)
        rootLinearVelocity,_ = pybullet.getBaseVelocity(self.humanoid)
        rightHandPosition, _, _, _, _, _ = pybullet.getLinkState(self.humanoid, self.terminalLinksIds[0])
        leftHandPosition, _, _, _, _, _ = pybullet.getLinkState(self.humanoid, self.terminalLinksIds[1])
        rightFootPosition, _, _, _, _, _ = pybullet.getLinkState(self.humanoid, self.terminalLinksIds[2])
        leftFootPosition, _, _, _, _, _ = pybullet.getLinkState(self.humanoid, self.terminalLinksIds[3])
        state = {
            "isTerminal" : rootPosition[2]<0.3 or rootPosition[2]>5,
            "jointsAngles": [], 
            "jointsVelocities": [],
            "rootPosition": list(rootPosition),
            "rootOrientation" : list(rootOrientation),
            "rootLinearVelocity" : list(rootLinearVelocity)[:2],
            "leftHandPosition" : list(leftHandPosition),
            "rightHandPosition" : list(rightHandPosition),
            "leftFootPosition" : list(leftFootPosition),
            "rightFootPosition" : list(rightFootPosition),
        }
        for s in jointStates:
            state["jointsAngles"].append([s[0]])
            state["jointsVelocities"].append([s[1]])

        return state  

    def compute_reward(self, state, reference_motion):
        if state["isTerminal"]:
            return 0
        w_p = 0.6
        w_v = 0.1
        w_e = 0.0
        w_c = 0.0

        scale_p = 2
        scale_v = 0.1
        scale_e = 40
        scale_c = 10

        postions = np.array(state["jointsAngles"])
        ref_postions = np.array(reference_motion["jointsAngles"])
        r_p = w_p*np.exp(-scale_p*np.sum((postions - ref_postions)**2))

        velocities = np.array(state["jointsVelocities"])
        ref_velocities = np.array(reference_motion["jointsVelocities"])
        r_v = w_v*np.exp(-scale_v*np.sum((velocities - ref_velocities)**2))

        error_e = np.sum((np.array(state["leftHandPosition"]) - np.array(reference_motion["leftHandPosition"]))**2)
        error_e += np.sum((np.array(state["rightHandPosition"]) - np.array(reference_motion["rightHandPosition"]))**2)
        error_e += np.sum((np.array(state["leftFootPosition"]) - np.array(reference_motion["leftFootPosition"]))**2)
        error_e += np.sum((np.array(state["rightFootPosition"]) - np.array(reference_motion["rightFootPosition"]))**2)
        r_e = w_e*np.exp(-scale_e*error_e)

        error_c = np.sum((np.array(state["rootPosition"]) - np.array(reference_motion["rootPosition"]))**2)
        r_c = w_c*np.exp(-scale_c*error_c)

        v = state["rootLinearVelocity"]
        vp = np.sqrt(v[0]**2 + v[1]**2)
        vp = min(vp, 4)/4
        r_vp = 0.2*vp

        return r_p + r_v + r_e + r_c + r_vp + 0.1


if __name__ == "__main__":
    path_to_data = "data/walking.json"
    with open(path_to_data, "r") as f:
        data = json.loads(f.read())
    env = RLEnv(initial_state=data["frames"][0])
    while True:
        1