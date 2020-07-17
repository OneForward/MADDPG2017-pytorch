import argparse
import numpy as np
import time
import pickle
import torch
from maddpg import MADDPGAgentTrainer
from matplotlib import pyplot as plt 
from path import Path 

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--q_lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--pi_lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="vis/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_agents(env, num_adversaries, arglist, agent = MADDPGAgentTrainer):
    obs_space_n, act_space_n = env.observation_space, env.action_space

    agents = []
    for idx in range(num_adversaries):
        agents.append(
            agent(obs_space_n, act_space_n, "agent_%d" % idx, idx, arglist,
                use_ddpg=(arglist.adv_policy=='ddpg')))
    for idx in range(num_adversaries, env.n):
        agents.append(
            agent(obs_space_n, act_space_n, "agent_%d" % idx, idx, arglist,
                use_ddpg=(arglist.adv_policy=='ddpg')))
    return agents

def save_state(agents, args):
    state = [{'pi':agent.pi.state_dict(), 
             'q':agent.q.state_dict(),
             'pi_targ':agent.pi_targ.state_dict(),
             'q_targ':agent.q_targ.state_dict(),
            } for agent in agents]
    
    torch.save(state, args.save_path / args.saved_filename)

def load_state(agents, args):
    states = torch.load(args.load_path / args.saved_filename)
    for agent, state in zip(agents, states):
        agent.pi.load_state_dict(state['pi'])
        agent.q.load_state_dict(state['q'])
        agent.pi_targ.load_state_dict(state['pi_targ'])
        agent.q_targ.load_state_dict(state['q_targ'])


def train(arglist):

    args.saved_filename = f"{args.good_policy}_v_{args.adv_policy}.pth"
    args.save_path = Path(args.save_dir) / args.scenario 
    args.load_path = Path(args.load_dir) / args.scenario 
    if not args.save_path.exists():
        args.save_path.makedirs()
    args.exp_name = f"{args.scenario}_{args.good_policy}_v_{args.adv_policy}"


    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    # Create agent agents
    num_adversaries = min(env.n, arglist.num_adversaries)
    agents = get_agents(env, num_adversaries, arglist)
    print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))


    # load_state Unimplemented in Pytorch Version 
    if arglist.display or arglist.restore or arglist.benchmark:
        print('Loading previous state...')
        load_state(agents, arglist)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    

    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        action_n = [agent.action(obs) for agent, obs in zip(agents,obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        for i, agent in enumerate(agents):
            agent.store(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all agents, if not in display or benchmark mode
        for agent in agents:
            agent.update(agents, train_step)

        # save model, display training output
        np.set_printoptions(precision=3)
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            save_state(agents, arglist)

            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print(f"steps: {train_step}, episodes: {len(episode_rewards)}, " 
                      f"mean episode reward: {np.mean(episode_rewards[-arglist.save_rate:]):.3f}, "
                      f"time: {round(time.time()-t_start, 3)}"
                     )
            else:
                print(f"\nsteps: {train_step}, episodes: {len(episode_rewards)}, " 
                      f"mean reward: {np.mean(episode_rewards[-arglist.save_rate:]):.3f}, "
                      f"\nagent episode reward: {np.array([np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards])}, "
                      f"time: {round(time.time()-t_start, 3)}"
                     )
                    
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            
            f_ep_rewards = arglist.plots_dir / f'{arglist.exp_name}_rewards.pkl'
            f_ep_ag_rewards = arglist.plots_dir / f'{arglist.exp_name}_agent_rewards.pkl'
            torch.save(final_ep_rewards, f_ep_rewards)
            torch.save(final_ep_ag_rewards, f_ep_ag_rewards)
            
        if len(episode_rewards) > arglist.num_episodes:
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    args = parse_args()

    args.plots_dir = Path(args.plots_dir)
    if not args.plots_dir.exists():
        args.plots_dir.makedirs()
        
    for s in ['simple_adversary', 'simple_tag', 'simple_crypto']:
        args.scenario = s
        for ag in ['maddpg', 'ddpg']:
            args.good_policy = ag
            for av in ['maddpg', 'ddpg']:
                args.adv_policy = av
                print (args.scenario, args.good_policy, args.adv_policy)
                train(args)
