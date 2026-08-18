"""
Real trained DQN baseline, rebuilt against the CORRECTED v3 congestion model
(genuine M/M/1-style within-batch congestion, not the broken static-queue
model). Reward mirrors PIEL's L (squared latency) + C (energy) + deadline
penalty terms; D(x) (entropy) is a global/population statistic and is
omitted from the per-step reward, same simplification as before.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os, time, pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIEL_pipeline_v3 import (load_config, load_rsu_profiles, load_tasks,
                               compute_T_batch, compute_avg_service_time,
                               latency_vec, energy_vec, load_entropy)

DATA_DIR = 'data'
cfg = load_config(os.path.join(DATA_DIR, 'experiment_config.csv'))
rsu = load_rsu_profiles(os.path.join(DATA_DIR, 'rsu_profiles.csv'))
N_RSU = rsu['n']
DEADLINE = cfg['deadline_ms']
ALPHA, BETA = cfg['alpha'], cfg['beta']

CAP_MAX = rsu['cap'].max()
WATT_MAX = rsu['watts'].max()


def gen_tasks(n, seed):
    rng = np.random.default_rng(seed)
    return {'size': rng.uniform(0.2, 2.0, n), 'cycles': rng.uniform(0.5e9, 2.0e9, n)}


# ============================================================================
#  Environment
# ============================================================================
class VECEnv:
    def __init__(self, n_tasks=300):
        self.n_tasks = n_tasks

    def reset(self, load_factor=None, seed=None):
        if load_factor is None:
            load_factor = np.random.uniform(0.15, 0.95)
        if seed is None:
            seed = np.random.randint(0, 10_000_000)
        self.tasks = gen_tasks(self.n_tasks, seed)
        self.T_batch = compute_T_batch(load_factor, self.tasks, rsu)
        self.avg_st  = compute_avg_service_time(self.tasks, rsu)
        self.busy_all = np.zeros(N_RSU)
        self.t = 0
        return self._state()

    def _state(self):
        i = self.t
        cap_n  = rsu['cap'] / CAP_MAX
        watt_n = rsu['watts'] / WATT_MAX
        rho_n  = np.minimum(self.busy_all / self.T_batch, 0.98)
        size_n = np.array([self.tasks['size'][i] / 2.0])
        cyc_n  = np.array([self.tasks['cycles'][i] / 2.0e9])
        return np.concatenate([cap_n, watt_n, rho_n, size_n, cyc_n]).astype(np.float32)

    def step(self, action):
        i = self.t
        lat = latency_vec(i, self.tasks, self.busy_all, rsu, cfg, self.T_batch, self.avg_st)[action]
        eng = energy_vec(i, self.tasks, self.busy_all, rsu, cfg, self.T_batch)[action]
        L = (lat / DEADLINE) ** 2
        C = eng / 50.0
        penalty = 10.0 if lat > DEADLINE else 0.0
        reward = -(ALPHA * L + BETA * C + penalty) / 5.0

        self.busy_all[action] += self.tasks['cycles'][i] / (rsu['cap'][action] * 1e9)
        self.t += 1
        done = self.t >= self.n_tasks
        next_state = self._state() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        info = {'latency': lat, 'energy': eng, 'success': lat <= DEADLINE}
        return next_state, reward, done, info


STATE_DIM = 3 * N_RSU + 2
ACTION_DIM = N_RSU


# ============================================================================
#  DQN (same architecture as before)
# ============================================================================
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = []; self.capacity = capacity; self.pos = 0
    def push(self, s, a, r, s2, d):
        item = (s, a, r, s2, d)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))
    def __len__(self):
        return len(self.buf)


def train_dqn(n_episodes=600, seed=0, verbose=False, gamma_rl=0.92,
              batch_size=128, lr=5e-4, target_sync=300, n_tasks=300,
              resume_path=None, ep_offset=0, total_episodes=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = VECEnv(n_tasks=n_tasks)
    qnet = QNet(STATE_DIM, ACTION_DIM)
    if resume_path and os.path.exists(resume_path):
        qnet.load_state_dict(torch.load(resume_path))
    target = QNet(STATE_DIM, ACTION_DIM)
    target.load_state_dict(qnet.state_dict())
    opt = optim.Adam(qnet.parameters(), lr=lr)
    buf = ReplayBuffer(capacity=50000)

    total_eps = total_episodes if total_episodes is not None else n_episodes
    eps_start, eps_end, eps_decay_episodes = 1.0, 0.05, int(total_eps * 0.7)
    step_count = 0
    episode_rewards = []

    for ep_local in range(n_episodes):
        ep = ep_local + ep_offset
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        s = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            if random.random() < eps:
                a = random.randrange(ACTION_DIM)
            else:
                with torch.no_grad():
                    qvals = qnet(torch.from_numpy(s).unsqueeze(0))
                    a = int(qvals.argmax(dim=1).item())
            s2, r, done, info = env.step(a)
            buf.push(s, a, r, s2, float(done))
            s = s2
            ep_reward += r
            step_count += 1

            if len(buf) >= batch_size * 4:
                bs, ba, br, bs2, bd = buf.sample(batch_size)
                bs_t  = torch.from_numpy(bs)
                bs2_t = torch.from_numpy(bs2)
                ba_t  = torch.from_numpy(ba).long()
                br_t  = torch.from_numpy(br)
                bd_t  = torch.from_numpy(bd)

                q_pred = qnet(bs_t).gather(1, ba_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(bs2_t).max(dim=1)[0]
                    q_target = br_t + gamma_rl * (1 - bd_t) * q_next
                loss = nn.functional.smooth_l1_loss(q_pred, q_target)
                opt.zero_grad(); loss.backward(); opt.step()

            if step_count % target_sync == 0:
                target.load_state_dict(qnet.state_dict())

        episode_rewards.append(ep_reward)
        if verbose and (ep_local + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            print(f"    episode {ep+1}/{total_eps}  eps={eps:.3f}  avg_reward(last50)={avg_r:.3f}")

    return qnet, episode_rewards


def evaluate_dqn(qnet, tasks, load_factor, n_tasks=300):
    """Greedy (argmax) evaluation of a trained DQN on a fixed task/load scenario."""
    T_batch = compute_T_batch(load_factor, tasks, rsu)
    avg_st  = compute_avg_service_time(tasks, rsu)
    busy_all = np.zeros(N_RSU)
    assignments = np.zeros(n_tasks, dtype=int)
    lat_arr = np.zeros(n_tasks); eng_arr = np.zeros(n_tasks)
    for i in range(n_tasks):
        cap_n  = rsu['cap'] / CAP_MAX
        watt_n = rsu['watts'] / WATT_MAX
        rho_n  = np.minimum(busy_all / T_batch, 0.98)
        size_n = np.array([tasks['size'][i] / 2.0])
        cyc_n  = np.array([tasks['cycles'][i] / 2.0e9])
        state = np.concatenate([cap_n, watt_n, rho_n, size_n, cyc_n]).astype(np.float32)
        with torch.no_grad():
            qvals = qnet(torch.from_numpy(state).unsqueeze(0))
            a = int(qvals.argmax(dim=1).item())
        lat = latency_vec(i, tasks, busy_all, rsu, cfg, T_batch, avg_st)[a]
        eng = energy_vec(i, tasks, busy_all, rsu, cfg, T_batch)[a]
        assignments[i] = a
        lat_arr[i] = lat; eng_arr[i] = eng
        busy_all[a] += tasks['cycles'][i] / (rsu['cap'][a] * 1e9)
    return {
        'assignments': assignments,
        'latency': float(lat_arr.mean()),
        'energy': float(eng_arr.mean()),
        'success': float((lat_arr <= DEADLINE).mean() * 100),
    }


def fair_score(qnet, cfg, rsu):
    """Evaluate on the 3 official scenarios using the SAME uniform evaluate()
    used for PIEL/Greedy/FIFO, for apples-to-apples checkpoint selection."""
    from PIEL_pipeline_v3 import load_tasks, evaluate
    total_success = 0.0
    total_latency = 0.0
    details = {}
    for scenario, lf in [('low_20', 0.20), ('moderate_50', 0.50), ('high_90', 0.90)]:
        tasks = load_tasks('data/tasks.csv', scenario_filter=scenario)
        res = evaluate_dqn(qnet, tasks, lf, n_tasks=300)
        T_batch = compute_T_batch(lf, tasks, rsu)
        avg_st = compute_avg_service_time(tasks, rsu)
        fair = evaluate(res['assignments'], tasks, rsu, cfg, T_batch, avg_st)
        details[scenario] = fair
        total_success += fair['success']
        total_latency += fair['latency']
    return total_success / 3.0, total_latency / 3.0, details


if __name__ == '__main__':
    print("Sanity check: v3 environment + training loop wired correctly.")
    t0 = time.time()
    qnet, rewards = train_dqn(n_episodes=20, verbose=True)
    print(f"20-episode smoke test finished in {time.time()-t0:.1f}s")
    print("First 5 rewards:", rewards[:5])
