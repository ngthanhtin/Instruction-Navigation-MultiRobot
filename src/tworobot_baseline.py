#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import math
#import gym
import numpy as np
import tensorflow as tf
# from ddpg import *
from mddpg.magent import *
from tworobot_environment import Env
from pathlib import Path
import argparse

import os
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help='1 for training and 0 for testing')
    parser.add_argument('--env_id', type=int, default=2, help='env name')
    parser.add_argument('--sac', type=int, default=0, help='1 for using sac')
    parser.add_argument('--visual_obs', type=int, default=0, help='1 for using image at robot observation')
    parser.add_argument('--test_env_id', type=int, default=2, help='test environment id')
    parser.add_argument('--n_scan', type=int, default=10, help='num of scan sampled from full scan')

    args = parser.parse_args()
    return args

def main():
    rospy.init_node('baseline')

    # get arg
    args = parse_args()
    is_training = bool(args.train)
    env_name = 'env' + str(args.env_id)
    trained_models_dir = './src/trained_models/bl-' + env_name + '-models/' if not args.visual_obs else \
            './src/trained_models/vis_obs-' + env_name + '-models/'

    env = Env(is_training, args.env_id, args.test_env_id, args.visual_obs, args.n_scan)
    
    # agent = DDPG(env, state_dim, action_dim, trained_models_dir)
    lr_actor = 1e-4
    lr_critic = 1e-4
    lr_decay = .95
    replay_buff_size = 10000
    gamma = .99
    batch_size = 128
    random_seed = 42
    soft_update_tau = 1e-3

    # 2 agents
    agent = MADDPG(state_dim, action_dim, lr_actor, lr_critic, lr_decay, replay_buff_size, gamma, batch_size, random_seed, soft_update_tau)

    past_action = np.array([[0., 0.], [0., 0.]])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    if is_training:
        print('Training mode')
        # path things
        figures_path = './figures/bl-' + env_name + '/' if not args.visual_obs else \
            './figures/vis_obs-' + env_name + '/'
        print(figures_path)
        Path(trained_models_dir + 'actor').mkdir(parents=True, exist_ok=True)
        Path(trained_models_dir + 'critic').mkdir(parents=True, exist_ok=True)
        Path(figures_path).mkdir(parents=True, exist_ok=True)

        avg_reward_his = []
        total_reward = 0
        var = 1.
        ep_rets = []
        ep_ret = 0.
        
        while True:
            states = env.reset()
            one_round_step = 0

            while True:
                a = agent.act(states)
                a[0][0] = np.clip(np.random.normal(a[0][0], var), 0., 1.)
                a[0][1] = np.clip(np.random.normal(a[0][1], var), -0.5, 0.5)
                a[1][0] = np.clip(np.random.normal(a[1][0], var), 0., 1.)
                a[1][1] = np.clip(np.random.normal(a[1][1], var), -0.5, 0.5)

                state_s, r, dones, arrives = env.step([a[0], a[1]], [past_action[0], past_action[1]])

                time_step = agent.update(states, a, r, state_s, dones)

                if arrives:
                    result = 'Success'
                else:
                    result = 'Fail'

                # if time_step > 0:
                #     total_reward += r
                #     ep_ret += r
                # print("Timestep: ",time_step)
                # if time_step % 10000 == 0 and time_step > 0:
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
                #     print('---------------------------------------------------')

                # if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                #     var *= 0.9999

                past_action = a
                states = state_s
                one_round_step += 1
                print(one_round_step)
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

    else:
        print('Testing mode')
        total_return = 0.
        total_step = 0
        total_path_len = 0.
        arrive_cnt = 0
        robot_name='turtlebot3_burger_1'
        # robot_name = 'robot1'
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
     main()
