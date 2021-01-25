from env import RLEnv
from networks import PNet, VNet
import torch
import torch.optim as optim
import json
import random
import numpy as np

class Buffer:
    def __init__(self):
        self.data = [[]]

    def __len__(self):
        l = 0
        for trajectory in self.data:
            l += len(trajectory)
        return l

    def append(self, d, new_trajectory=False):
        if new_trajectory:
            self.data.append([d])
        else:
            self.data[-1].append(d)

    def get_data(self):
        # flatten the data
        return sum(self.data, [])


class RLBuffer:
    def __init__(self):
        self.states = Buffer()
        self.actions = Buffer()
        self.rewards = Buffer()
        self.log_probs = Buffer()
        self.values = Buffer()
        self.advantages = Buffer()
    
    def __len__(self):
        return len(self.values)

    def append(self, state, action, reward, log_prob, value, new_trajectory=False):
        self.states.append(state, new_trajectory)
        self.actions.append(action, new_trajectory)
        self.rewards.append(reward, new_trajectory)
        self.log_probs.append(log_prob, new_trajectory)
        self.values.append(value, new_trajectory)

    def apply_GAE(self, Gamma=0.95, Lambda=0.95):
        self.advantages.data = []
        for i in range(len(self.values.data)):
            advantages = GAE(self.values.data[i], self.rewards.data[i], Gamma=Gamma, Lambda=Lambda)
            self.advantages.data.append(advantages)

    def get_data(self):
        return (self.states.get_data(), self.actions.get_data(), self.rewards.get_data(), 
            self.log_probs.get_data(), self.values.get_data(), self.advantages.get_data())


def append_to_file(data, path): 
    with open(path, 'a') as outfile:
        outfile.write(json.dumps(data))
        outfile.write(",")
        outfile.close() 


def GAE(values, rewards, Gamma=0.95, Lambda=0.95):
    T = len(rewards)
    A = rewards[T-1] - values[T-1]
    advantages = [A]
    for t in reversed(range(T-1)):
        delta = rewards[t] + Gamma * values[t + 1]  - values[t]
        A = delta + Gamma * Lambda * A
        advantages.insert(0, A)
    return advantages


def test(reference_motion, n_episodes=10):
    frames = reference_motion["frames"]
    n_frames = len(frames)
    
    rewards = []
    with torch.no_grad(): 
        for _ in range(n_episodes):
            state = env.reset(frames[0])
            total_reward = 0
            for t in range(n_frames):
                state = state_to_tensor(state)
                policy = Pmodel(state)
                action = policy.sample()
                next_state, reward = env.step(action.cpu().numpy()[0,:], frames[t])
                total_reward += reward
                if next_state["isTerminal"]:
                    break
                else:
                    state = next_state
            rewards.append(total_reward)
    return np.mean(rewards)


def state_to_tensor(state):
    angles = list(np.array(state["jointsAngles"])[:,0])
    velocities = list(np.array(state["jointsVelocities"])[:,0])
    T = torch.FloatTensor([
        angles+velocities
        +state["rootPosition"]+state["rootOrientation"]+state["rootLinearVelocity"]
        +state["leftHandPosition"]+state["rightHandPosition"]
        +state["leftFootPosition"]+state["rightFootPosition"]
    ]).to(device)
    return T


def train(reference_motion, Gamma=0.95, Lambda=0.95, n_episodes=1000, n_steps=500, minibatch_size=256, update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10):
    frames = reference_motion["frames"]
    n_frames = len(frames)
    best_reward = 10.0

    for episode in range(n_episodes):
        buffer = RLBuffer()
        new_trajectory = False
        while len(buffer) < update_every:
            t0 = random.choice(range(n_frames//2))
            state = env.reset(frames[t0])
            for t in range(t0, min(t0 + n_steps, n_frames)):
                state = state_to_tensor(state)
                
                policy = Pmodel(state)
                value = Vmodel(state)
                
                action = policy.sample()
                next_state, reward = env.step(action.cpu().numpy()[0,:], frames[t])
                log_prob = policy.log_prob(action)

                buffer.append(state, action, torch.FloatTensor([reward]).to(device), log_prob, value, new_trajectory=new_trajectory)
                append_to_file(reward, path="data/allrewards.txt")
        
                if next_state["isTerminal"]:
                    new_trajectory = True
                    break
                else:
                    state = next_state
                    new_trajectory = False

        buffer.apply_GAE()
        states, actions, rewards, log_probs, values, advantages_GAE = buffer.get_data()
                
        advantages_GAE = torch.cat(advantages_GAE).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        
        values_TD = advantages_GAE + values
        
        PPO(advantages_GAE, log_probs, values_TD, states, actions, minibatch_size=minibatch_size, n_updates=n_updates, epsilon=epsilon)
        
        if episode % test_every == 0:
            avg_test_reward = test(reference_motion, test_episodes)
            print(f"episode {episode}, avg test reward {avg_test_reward}")
            append_to_file(avg_test_reward, path="data/testrewards.txt")
            avg_test_reward = round(avg_test_reward, 2)
            if avg_test_reward > best_reward:
                torch.save(Pmodel.state_dict(), f"data/policy_{avg_test_reward}_ep_{episode}.pth")
                torch.save(Vmodel.state_dict(), f"data/value_function_{avg_test_reward}_ep_{episode}.pth")
                best_reward = avg_test_reward


def PPO(advantages_GAE, log_probs, values_TD, states, actions, minibatch_size=256, n_updates=20, epsilon=0.2, pg=True):
    for _ in range(n_updates):
        rand_ids = np.random.randint(0, len(states), minibatch_size)
        A_GAE = advantages_GAE[rand_ids,:]
        lp = log_probs[rand_ids]
        V_TD = values_TD[rand_ids,:]
        s = states[rand_ids,:]
        a = actions[rand_ids,:]
        
        new_policies = Pmodel(s)
        new_values = Vmodel(s)
        
        new_lp = new_policies.log_prob(a)
        
        Poptimizer.zero_grad()

        if pg:
            policy_loss  = - (new_lp * A_GAE).mean()
        else:
            W = (new_lp - lp).exp()
            term1 = W * A_GAE
            term2 = torch.clamp(W, 1.0 - epsilon, 1.0 + epsilon) * A_GAE
            policy_loss  = - torch.min(term1, term2).mean()
        
        policy_loss.backward()
        Poptimizer.step()
        
        Voptimizer.zero_grad()
        value_loss = (V_TD - new_values).pow(2).mean()
        value_loss.backward()
        Voptimizer.step()


def evaluate_policy(policy_path, value_function_path):
    policy_state_dict = torch.load(policy_path)
    Pmodel = PNet(dim_state, dim_action, scale=0.01)
    Pmodel.load_state_dict(policy_state_dict)
    Pmodel.to(device)

    value_state_dict = torch.load(value_function_path)
    Vmodel = VNet(dim_state)
    Vmodel.load_state_dict(policy_state_dict)
    Vmodel.to(device)
    # TODO
    return 


if __name__ == "__main__":
    path_to_data = "data/walking.json"
    policy_path = None
    value_path = None

    with open(path_to_data, "r") as f:
        reference_motion = json.loads(f.read())

    env = RLEnv(useGUI=True, timeStep=reference_motion["timestep"])
    dim_state = env.dim_state()
    dim_action = env.dim_action()
    print(f"dim_action={dim_action}")
    print(f"dim_state={dim_state}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"running on {device}")

    Pmodel = PNet(dim_state, dim_action, scale=0.05) # the scale should be small 
    if policy_path:
        policy_state_dict = torch.load(policy_path)
        Pmodel.load_state_dict(policy_state_dict)
    Pmodel.to(device)
    Poptimizer = optim.Adam(Pmodel.parameters(), lr=0.001)

    Vmodel = VNet(dim_state).to(device)
    if value_path:
        value_state_dict = torch.load(value_path)
        Vmodel.load_state_dict(value_state_dict)
    Vmodel.to(device)
    Voptimizer = optim.Adam(Vmodel.parameters(), lr=0.01)

    train(reference_motion, Gamma=0.95, Lambda=0.95, n_episodes=20000, n_steps=500, minibatch_size=256,
            update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10)

    torch.save(Pmodel.state_dict(), "data/policy_2.pth")
    torch.save(Vmodel.state_dict(), "data/value_function_2.pth")
