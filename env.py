import pybullet
import time
import pybullet_data

class RLEnv:
    def __init__(self, initial_state=None, useGUI=True, timeStep=1/30):

        self.useGUI = useGUI
        if self.useGUI:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        pybullet.setPhysicsEngineParameter(fixedTimeStep=timeStep, numSubSteps=1)
        pybullet.setGravity(0,0,-9.81)
        self.humanoid = None
        self.jointIds = None
        self.jointNames = None
        self.reset(initial_state)
        return

    def reset(self, initial_state=None):
        pybullet.resetSimulation()
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
            pybullet.resetJointStateMultiDof(self.humanoid, self.jointIds, initial_state["jointsAngles"], initial_state["jointsVelocities"])
            return initial_state
        else:
            return self.get_state()

    def n_actions(self):
        return len(self.jointIds)

    def step(self, action, reference_motion):
        pybullet.setJointMotorControlArray(self.humanoid, self.jointIds, pybullet.POSITION_CONTROL, action)
        pybullet.stepSimulation()
        next_state = self.get_state()
        reward = self.compute_reward(next_state, reference_motion)
        return next_state, reward
        
    def compute_reward(self, state, reference_motion):
        #TODO
        return 1

    def get_state(self):
        jointStates = pybullet.getJointStates(self.humanoid, self.jointIds)
        rootPosition, rootOrientation = pybullet.getBasePositionAndOrientation(self.humanoid)
        state = {
            "isTerminal" : False, #TODO check body positions
            "jointsAngles": [], 
            "jointsVelocities": [],
            "rootPosition": rootPosition,
            "rootOrientation" : rootOrientation,
            "leftHandPosition" : (0,0,0), #TODO use (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
            "rightHandPosition" : (0,0,0), #TODO
            "leftFootPosition" : (0,0,0), #TODO
            "rightFootPosition" : (0,0,0), #TODO
        }
        for s in jointStates:
            state["jointsAngles"].append(s[0])
            state["jointsVelocities"].append(s[1])

        return state  


if __name__ == "__main__":
    env = RLEnv()