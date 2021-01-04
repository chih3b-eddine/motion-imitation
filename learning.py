from env import RLEnv
from networks import PNet, VNet
import torch
import torch.optim as optim
import json
import random
import numpy as np


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
            state = frames[0]
            env.reset(state)
            total_reward = 0
            for t in range(n_frames):
                state = state_to_tensor(state)
                policy = Pmodel(state)
                value = Vmodel(state)
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
        +state["rootPosition"]+state["rootOrientation"]
        +state["leftHandPosition"]+state["rightHandPosition"]
        +state["leftFootPosition"]+state["rightFootPosition"]
    ]).to(device)
    return T


def train(reference_motion, Gamma=0.95, Lambda=0.95, n_episodes=1000, n_steps=500, minibatch_size=256, update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10):
    frames = reference_motion["frames"]
    n_frames = len(frames)

    for episode in range(n_episodes):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        while len(values) < update_every:
            t0 = random.choice(range(n_frames//2))
            state = frames[t0]
            env.reset(state)
            for t in range(t0, min(t0 + n_steps, n_frames)):
                state = state_to_tensor(state)
                
                policy = Pmodel(state)
                value = Vmodel(state)
                
                action = policy.sample()
                next_state, reward = env.step(action.cpu().numpy()[0,:], frames[t])
                log_prob = policy.log_prob(action)
                
                log_probs.append(log_prob)
                values.append(value)
                states.append(state)
                actions.append(action)
                rewards.append(torch.FloatTensor([reward]).to(device))
        
                if next_state["isTerminal"]:
                    break
                else:
                    state = next_state
                
        advantages_GAE = GAE(values, rewards, Gamma=Gamma, Lambda=Lambda)
        
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


if __name__ == "__main__":
    path_to_data = "data/walking.json"

    with open(path_to_data, "r") as f:
        reference_motion = json.loads(f.read())

    env = RLEnv(useGUI=True, timeStep=reference_motion["timestep"])
    dim_state = env.dim_state()
    dim_action = env.dim_action()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"running on {device}")

    Pmodel = PNet(dim_state, dim_action, scale=1).to(device)
    Poptimizer = optim.Adam(Pmodel.parameters(), lr=0.001)

    Vmodel = VNet(dim_state).to(device)
    Voptimizer = optim.Adam(Vmodel.parameters(), lr=0.01)

    train(reference_motion, Gamma=0.95, Lambda=0.95, n_episodes=1000, n_steps=500, minibatch_size=256,
            update_every=4096, n_updates=20, epsilon=0.2, test_every=5, test_episodes=10)

    torch.save(Pmodel.state_dict(), "data/policy.pth")
    torch.save(Vmodel.state_dict(), "data/value_function.pth")
