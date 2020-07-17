import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from core import *

class MADDPGAgentTrainer():
    def __init__(self, obs_space_n, act_space_n, name, idx, args, 
                use_ddpg=False, hidden_sizes=(64,64), activation=nn.ReLU, 
                polyak = 1.0 - 1e-2,
                replay_size=1e6, update_every=100
            ):
        self.name = name
        self.idx = idx
        self.args = args
        self.polyak = polyak
        self.update_every = update_every
        self.update_after = args.batch_size * args.max_episode_len
        self.batch_size = args.batch_size
        self.use_ddpg = use_ddpg

        obs_dim = obs_space_n[idx].shape[0]
        act_dim = act_space_n[idx].n

        obs_dim_total = sum(obs_space.shape[0] for obs_space in obs_space_n)
        act_dim_total = sum(act_space.n for act_space in act_space_n)
         
        # Create actor-critic module and target networks
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)        
        self.pi_targ = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        
        if use_ddpg:
            self.q  = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
            self.q_targ  = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.q  = MLPQFunction(obs_dim_total, act_dim_total, hidden_sizes, activation)
            self.q_targ  = MLPQFunction(obs_dim_total, act_dim_total, hidden_sizes, activation)

        self.act_dim = act_dim
        self.n = len(obs_space_n)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.pi_targ.parameters():
            p.requires_grad = False
        for p in self.q_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi.parameters(), lr=self.args.pi_lr)
        self.q_optimizer = Adam(self.q.parameters(), lr=self.args.q_lr)

        self.buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size))

    def action(self, o, noise_scale=.3):
        o = torch.as_tensor(o, dtype=torch.float32)
        with torch.no_grad():
            p = self.pi(o)
            # p += noise_scale*torch.randn_like(p)
            return gumbel_softmax(p).numpy()

    def store(self, o, a, r, o2, d):
        # Store transition in the replay buffer.
        self.buffer.store(o, a, r, o2, d)

    def update(self, agents, t):
        # self.update_after = 100
        if t < self.update_after or not t % self.update_every == 0:  # only update every 100 steps
            return

        # collect replay sample from all agents
        idxs = self.buffer.make_index(self.batch_size)
        o_n, o2_n, a_n = [], [] , []
        for agent in agents:
            o, o2, a, r, d = agent.buffer.sample_index(idxs)
            o_n.append(o)
            o2_n.append(o2)
            a_n.append(a)
        o, o2, a, r, d  = self.buffer.sample_index(idxs)


        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.q.parameters():
            p.requires_grad = True

        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(agents, r, d, o_n, a_n, o2_n)
        loss_q.backward()
        clip_grad_norm_(self.q.parameters(), 0.5)
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.q.parameters():
            p.requires_grad = False


        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(o_n, a_n)
        loss_pi.backward()
        clip_grad_norm_(self.pi.parameters(), 0.5)
        self.pi_optimizer.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.pi.parameters(), self.pi_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_q, loss_pi

    # Set up function for computing MADDPG Q-loss
    def compute_loss_q(self, agents, r, d, o_n, a_n, o2_n):
        if self.use_ddpg:
            q = self.q([o_n[self.idx]], [a_n[self.idx]])
        else:
            q = self.q(o_n, a_n)

        with torch.no_grad():
            # Bellman backup for Q function
            a_n_targ = [ gumbel_softmax(agent.pi_targ(o2)) 
                        for agent, o2 in zip(agents, o2_n) ]
            if self.use_ddpg:
                q_targ = self.q_targ([o2_n[self.idx]], [a_n_targ[self.idx]])
            else:
                q_targ = self.q_targ(o2_n, a_n_targ)
            
            y = r + self.args.gamma * (1.0 - d) * q_targ

        # MSE loss against Bellman backup
        loss_q = ((q - y)**2).mean()

        return loss_q

    # Set up function for computing MADDPG pi loss
    def compute_loss_pi(self, o_n, a_n):
 
        p = self.pi(o_n[self.idx])
        p_reg = (p**2).mean()

        a_n[self.idx] =  gumbel_softmax(p)
        
        if self.use_ddpg:
            q = self.q([o_n[self.idx]], [a_n[self.idx]])
        else:
            q = self.q(o_n, a_n)
        loss_pi = -q.mean()
        return loss_pi + p_reg * 1e-3


