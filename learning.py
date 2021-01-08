from env import RLEnv
from networks import PNet, VNet, StateDist, PhaseDist
import torch
import torch.optim as optim
import json
import random
import numpy as np
import os


def GAE(values, rewards, Gamma=0.95, Lambda=0.95):
    T = len(rewards)
    A = rewards[T-1] - values[T-1]
    advantages = [A]
    for t in reversed(range(T-1)):
        delta = rewards[t] + Gamma * values[t + 1]  - values[t]
        A = delta + Gamma * Lambda * A
        advantages.insert(0, A)
    return advantages


def test(frames, n_episodes=10):
    n_frames = len(frames)
    
    rewards = []
    with torch.no_grad(): 
        for _ in range(n_episodes):
            state = frames[0]
            env.reset(state)
            state = extend_state(state_to_tensor(state), [0.0])
            total_reward = 0
            for t in range(n_frames):
                policy = Pmodel(state)
                value = Vmodel(state)
                index, _ = stateDist.evaluate_state(state[0][:-1])
                phase = phase_values[index]
                action = policy.sample()
                next_state, reward = env.step(action.cpu().numpy()[0,:], tensor_to_state(state))
                total_reward += reward
                if next_state["isTerminal"]:
                    break
                else:
                    state = extend_state(state_to_tensor(next_state), [phase]) 
            rewards.append(total_reward)
    return np.mean(rewards)


def state_to_tensor(state):
    angles = list(np.array(state["jointsAngles"])[:,0])
    velocities = list(np.array(state["jointsVelocities"])[:,0])
    T = torch.FloatTensor([
        angles+velocities
        +state["rootPosition"]+state["rootOrientation"]
        +state["leftHandPosition"]+state["rightHandPosition"]
        +state["leftFootPosition"]+state["rightFootPosition"]
    ]).to(device)
    return T


def tensor_to_state(T):
    state = {
        "jointsAngles" : [[i] for i in T[0,:21].tolist()],
        "jointsVelocities" :  [[i] for i in T[0,21:42].tolist()],
        "rootPosition" : T[:,42:45][0].tolist(),
        "rootOrientation" : T[:,45:49][0].tolist(),
        "leftHandPosition" : T[:,49:52][0].tolist(),
        "rightHandPosition" : T[:,52:55][0].tolist(),
        "leftFootPosition" : T[:,55:58][0].tolist(),
        "rightFootPosition" : T[:,58:61][0].tolist()
    }
    return state


def get_init_weights(frames, positions):
    X = torch.cat([state_to_tensor(frame) for frame in frames])
    means = X[positions]
    sigma = torch.FloatTensor(np.cov(X.T))
    return means, sigma


def extend_state(state, phase):
    phase = torch.FloatTensor([phase]).to(device)
    return torch.cat((state, phase), dim=-1)


def train(frames, Gamma=0.95, Lambda=0.95, n_episodes=1000, n_steps=500, minibatch_size=256, update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10):
    n_frames = len(frames)

    for episode in range(n_episodes):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        phases = []
        while len(values) < update_every:
            index0, phase0, t0 = phaseDist.sample()
            state = stateDist.sample(index0).unsqueeze(0)
            state = extend_state(state, [phase0])
            env.reset(tensor_to_state(state))
            for t in range(t0, min(t0 + n_steps, n_frames)):                
                policy = Pmodel(state)
                value = Vmodel(state)
                index, _ = stateDist.evaluate_state(state[0][:-1])
                phase = phase_values[index]
                
                action = policy.sample()
                next_state, reward = env.step(action.cpu().numpy()[0,:], tensor_to_state(state))
                log_prob = policy.log_prob(action)
                
                log_probs.append(log_prob)
                values.append(value)
                states.append(state)
                actions.append(action)
                rewards.append(torch.FloatTensor([reward]).to(device))
                phases.append(phase)
        
                if next_state["isTerminal"]:
                    break
                else:
                    state = extend_state(state_to_tensor(next_state), [phase])
                
        advantages_GAE = GAE(values, rewards, Gamma=Gamma, Lambda=Lambda)
        
        advantages_GAE = torch.cat(advantages_GAE).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        
        values_TD = advantages_GAE + values
        
        PPO(advantages_GAE, log_probs, values_TD, states, actions, minibatch_size=minibatch_size, n_updates=n_updates, epsilon=epsilon)
        
        if episode % test_every == 0:
            avg_test_reward = test(frames, test_episodes)
            print(f"episode {episode}, avg test reward {avg_test_reward}")
            avg_test_reward = round(avg_test_reward, 2)
            if avg_test_reward > 1.10:
                torch.save(Pmodel.state_dict(), f"data/policy_{avg_test_reward}_ep_{episode}.pth")
                torch.save(Vmodel.state_dict(), f"data/value_function_{avg_test_reward}_ep_{episode}.pth")



def PPO(advantages_GAE, log_probs, values_TD, states, actions, minibatch_size=256, n_updates=20, epsilon=0.2):
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
    policy_path = "data/policy.pth"
    value_path = "data/value_function.pth"

    with open(path_to_data, "r") as f:
        reference_motion = json.loads(f.read())

    env = RLEnv(useGUI=True, timeStep=reference_motion["timestep"])
    dim_state = env.dim_state()
    dim_action = env.dim_action()
    k = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"running on {device}")

    Pmodel = PNet(dim_state+1, dim_action, scale=0.01) # the scale should be small 
    if os.path.isfile(policy_path):
        policy_state_dict = torch.load(policy_path)
        Pmodel.load_state_dict(policy_state_dict)
    Pmodel.to(device)
    Poptimizer = optim.Adam(Pmodel.parameters(), lr=0.001)

    Vmodel = VNet(dim_state+1).to(device)
    if os.path.isfile(value_path):
        value_state_dict = torch.load(value_path)
        Vmodel.load_state_dict(value_state_dict)
    Vmodel.to(device)
    Voptimizer = optim.Adam(Vmodel.parameters(), lr=0.01)
    
    frames = reference_motion["frames"]
    phaseDist = PhaseDist(k)
    phase_values, positions = phaseDist.fit(len(frames))
    means, sigma = get_init_weights(frames, positions)
    stateDist = StateDist(dim_state, k, means, sigma)

    train(frames, Gamma=0.95, Lambda=0.95, n_episodes=1000, n_steps=500, minibatch_size=256,
            update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10)

    torch.save(Pmodel.state_dict(), "data/policy_2.pth")
    torch.save(Vmodel.state_dict(), "data/value_function_2.pth")
