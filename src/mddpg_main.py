#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import math
import numpy as np
from mddpg.mddpg_agent import *

from environments.multirobot_environment import Env
from pathlib import Path
import argparse, os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

exploration_decay_start_step = 50000
state_dim = 16
action_dim = 2
action_linear_max = 0.25  # m/s
action_angular_max = 0.5  # rad/s

def write_to_csv(item, file_name):
    with open(file_name, 'a') as f:
        f.write("%s\n" % item)

def seed_torch(seed):
    torch.manual_seed = seed
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train(args):
    np.random.seed(args.seed)
    seed_torch(args.seed)

    rospy.init_node('MDDPG')

    is_training = True

    trained_models_dir = './src/model_weights/mddpg/'
    if not os.path.isdir(trained_models_dir):
        os.makedirs(trained_models_dir)

    env = Env(is_training, args.env_id, args.test_env_id, args.num_agents, args.visual_obs, args.n_scan)

    # agents
    agent = MADDPG(state_dim, action_dim, args.num_agents, args.buffer_size, args.gamma, args.batch_size, args.seed, args.tau)

    past_action = np.array([[0., 0.], [0., 0.]])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    print('Training mode')

    threshold_init = 20
    worsen_tolerance = 0
    avg_reward_his = []
    total_reward = []
    avg_scores = []
    max_score = -1
    var = 1.
    ep_rets = []
    ep_ret = 0.

    while True:
        states = env.reset()
        one_round_step = 0
        scores = np.zeros(2) 
        while True:
            agent.total_step += 1
            a = agent.act(states)
            a[0][0] = np.clip(np.random.normal(a[0][0], var), 0., 1.)
            a[0][1] = np.clip(np.random.normal(a[0][1], var), -0.5, 0.5)
            a[1][0] = np.clip(np.random.normal(a[1][0], var), 0., 1.)
            a[1][1] = np.clip(np.random.normal(a[1][1], var), -0.5, 0.5)

            state_s, r, dones, arrives = env.step([a[0], a[1]], [past_action[0], past_action[1]])

            agent.update(states, a, r, state_s, dones)

            if arrives:
                result = 'Success'
            else:
                result = 'Fail'

            
            # if agent.total_step > 0:
            #     total_reward += r
            #     ep_ret += r
            # print("Timestep: ",agent.total_step)
            # if agent.total_step % 10000 == 0 and agent.total_step > 0:
            #     print('---------------------------------------------------')
            #     avg_reward = total_reward / 10000
            #     print('Average_reward = ', avg_reward)
            #     avg_reward_his.append(round(avg_reward, 2))
            #     print('Average Reward:',avg_reward_his)
            #     total_reward = 0
            #     print('Mean episode return over training time step: {:.2f}'.format(np.mean(ep_rets)))
            #     print('Mean episode return over current 10k training time step: {:.2f}'.format(np.mean(ep_rets[-10:])))
            #     write_to_csv(np.mean(ep_rets), figures_path + 'mean_ep_ret_his.csv')
            #     write_to_csv(np.mean(ep_rets[-10:]), figures_path + 'mean_ep_ret_10k_his.csv')
            #     write_to_csv(avg_reward, figures_path + 'avg_reward_his.csv')
                print('---------------------------------------------------')

            if agent.total_step % 5 == 0 and agent.total_step > exploration_decay_start_step:
                var *= 0.9999

            scores += np.array(r)
            past_action = a
            states = state_s
            one_round_step += 1

            # if arrive_s:
            #     print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
            #     one_round_step = 0
            #     if time_step > 0:
            #         ep_rets.append(ep_ret)
            #         ep_ret = 0.

            # if done_s or one_round_step >= 500:
            #     print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
            #     if time_step > 0:
            #         ep_rets.append(ep_ret)
            #         ep_ret = 0.
            #     break

            if (dones[0] == 1 and dones[1] == 1) or (arrives[0] == 1 and arrives[1] == 1) or one_round_step >= 500:
                break
        
        episode_score = np.max(scores)
        total_reward.append(episode_score)
        print("Score: {:.4f}".format(episode_score))

        if max_score <= episode_score:                     
            max_score = episode_score
            agent.save( trained_models_dir + './tworobot_weights.pth')

        if len(total_reward) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(total_reward[(len(total_reward)-100):]) / 100
            print("100 Episodic Everage Score: {:.4f}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)
        
            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = threshold_init           # re-count tolerance
                max_avg_score = latest_avg_score
            else:                                           
                if max_avg_score > 0.5:                     
                    worsen_tolerance -= 1                   # count worsening counts
                    print("Loaded from last best model.")
                    agent.load(trained_models_dir + '/tworobot_weights.pth')             # continue from last best-model
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break

def test(args):
    np.random.seed(args.seed)
    seed_torch(args.seed)

    print('Testing mode')
    total_return = 0.
    total_step = 0
    total_path_len = 0.
    arrive_cnt = 0
    robot_name='turtlebot3_burger_1'
    # robot_name = 'robot1'


    is_training = False
    env_name = 'env' + str(args.env_id)
    trained_models_dir = './src/trained_models/bl-' + env_name + '-models/' if not args.visual_obs else \
            './src/trained_models/vis_obs-' + env_name + '-models/'

    
    env = Env(is_training, args.env_id, args.test_env_id, args.num_agents, args.visual_obs, args.n_scan)

    # 2 agents
    agent = MADDPG(state_dim, action_dim, args.num_agents, args.buffer_size, args.gamma, args.batch_size, args.seed, args.tau)

    past_action = np.array([[0., 0.], [0., 0.]])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    while True:
        state = env.reset()
        
        one_round_step = 0

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5)
            except:
                pass

        robot_cur_state = data.pose[data.name.index(robot_name)].position
        
        while True:
            a = agent.action(state)
            a[0] = np.clip(a[0], 0., 1.)
            a[1] = np.clip(a[1], -0.5, 0.5)
            state_, r, done, arrive = env.step(a, past_action)
            total_return += r
            past_action = a
            state = state_
            one_round_step += 1
            total_step += 1

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5)
                except:
                    pass

            robot_next_state = data.pose[data.name.index(robot_name)].position
            dist = math.hypot(
                    robot_cur_state.x - robot_next_state.x,
                    robot_cur_state.y - robot_next_state.y
                    )
            total_path_len += dist
            robot_cur_state = robot_next_state

            if arrive:
                arrive_cnt += 1
                print('Step: %3i' % one_round_step, '| Arrive!!!')
                one_round_step = 0
                if env.test_goals_id >= len(env.test_goals):
                    print('Finished, total return: ', total_return)
                    print('Total step: ', total_step)
                    print('Total path length: ', total_path_len)
                    print('Success rate: ', arrive_cnt / len(env.test_goals))
                    exit(0)

            if done:
                print('Step: %3i' % one_round_step, '| Collision!!!')
                if env.test_goals_id >= len(env.test_goals):
                    print('Finished, total return: ', total_return)
                    print('Total step: ', total_step)
                    print('Total path length: ', total_path_len)
                    print('Success rate: ', arrive_cnt / len(env.test_goals))
                    exit(0)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help='1 for training and 0 for testing')
    parser.add_argument('--env_id', type=int, default=2, help='env name')
    parser.add_argument('--visual_obs', type=int, default=0, help='1 for using image at robot observation')
    parser.add_argument('--test_env_id', type=int, default=2, help='test environment id')
    parser.add_argument('--n_scan', type=int, default=10, help='num of scan sampled from full scan')
    
    #model parameters
    parser.add_argument('--num-agents', type=int, default=2, help='number of agents')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Size of the replay buffer')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--tau', type=float, default=1e-3, help='tau for soft update')
    parser.add_argument('--initial-random-steps', type=int, default=int(1e2), help="initital Exploration steps")
    parser.add_argument('--policy_update_fequency', type=int, default=2, help='policy update frequency')

    #device
    parser.add_argument('--device', type=str, default='cpu', help='Which devices to use, cuda or cpu')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    args = parser.parse_args()

    if args.train == 1:
        train(args)
    else:
        test(args)
