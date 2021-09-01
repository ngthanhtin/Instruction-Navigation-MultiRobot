#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import math
import numpy as np
from msac.msac_agent import *
from environments.multirobot_environment import Env
from pathlib import Path
import argparse, os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    rospy.init_node('MSAC')

    trained_models_dir = './src/model_weights/msac/'
    if not os.path.isdir(trained_models_dir):
        os.makedirs(trained_models_dir)

    is_training = True
    env = Env(is_training, args.env_id, args.test_env_id, args.num_agents, args.visual_obs, args.n_scan)
    
    replay_buff_size = args.buffer_size
    batch_size = args.batch_size
    gamma = args.gamma
    tau = args.tau
    initial_random_steps = args.initial_random_steps
    policy_update_frequency = args.policy_update_frequency
    num_agents = args.num_agents

    # 2 agents
    agent = MSAC(state_dim, action_dim, replay_buff_size, batch_size, gamma, tau, initial_random_steps, policy_update_frequency, num_agents)

    past_action = np.array([[0., 0.], [0., 0.]])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    print('Training mode')

    total_rewards = []
    avg_scores = []
    max_score = -1
    var = 1.

    actor_losses, qf_losses, v_losses, alpha_losses = [], [], [], []
    scores = []
    avg_scores = []
    score = [0 for _ in range(num_agents)]
    num_episode = 0
    while True:
        states = env.reset()
        score = np.zeros(num_agents) 
        num_episode += 1
        one_round_step = 0
        while True:
            agent.total_step += 1
            one_round_step += 1
            a = agent.select_action(states)
            a[0][0] = np.clip(np.random.normal(a[0][0], var), 0., 1.)
            a[0][1] = np.clip(np.random.normal(a[0][1], var), -0.5, 0.5)
            a[1][0] = np.clip(np.random.normal(a[1][0], var), 0., 1.)
            a[1][1] = np.clip(np.random.normal(a[1][1], var), -0.5, 0.5)

            state_s, r, dones, arrives = env.step([a[0], a[1]], [past_action[0], past_action[1]])

            for i in range(num_agents):
                agent.transition[i] += [r[i], state_s[i], dones[i]]
                agent.memory.store(*agent.transition[i])

            score += np.array(r)
            past_action = a
            states = state_s

            if (len(agent.memory) > batch_size and agent.total_step > initial_random_steps):
                loss = agent.update_model()
                actor_losses.append(loss[0])
                qf_losses.append(loss[1])
                v_losses.append(loss[2])
                alpha_losses.append(loss[3])
            
            if agent.total_step % 5 == 0 and agent.total_step > initial_random_steps:
                var *= 0.9999

            if np.any(arrives) or one_round_step >= 500:
                break

        scores.append(max(score))
        if num_episode >= 100:
            avg_scores.append(np.mean(scores[-100:]))
        # plot(num_episode, scores, avg_scores, actor_losses, qf_losses, v_losses, alpha_losses)


        episode_score = np.max(score)
        total_rewards.append(episode_score)
        print("Score: {:.4f}".format(episode_score))

        # if max_score <= episode_score:                     
        #     max_score = episode_score
        #     agent.save(trained_models_dir + '/tworobot_weights.pth')

        if len(total_rewards) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(total_rewards[(len(total_rewards)-100):]) / 100
            print("100 Episodic Average Score: {:.4f}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)
        

        torch.save(agent.actor.state_dict(), trained_models_dir + "/mactor.pt")
        torch.save(agent.qf1.state_dict(), trained_models_dir + "/mqf1.pt")
        torch.save(agent.qf2.state_dict(), trained_models_dir + "/mqf2.pt")
        torch.save(agent.vf.state_dict(), trained_models_dir + "/mvf.pt")

def test(args):
    np.random.seed(args.seed)
    seed_torch(args.seed)

    rospy.init_node('MSAC TESTING')

    is_training = False

    env = Env(is_training, args.env_id, args.test_env_id, args.num_agents, args.visual_obs, args.n_scan)
    
    
    replay_buff_size = args.buffer_size
    batch_size = args.batch_size
    gamma = args.gamma
    tau = args.tau
    initial_random_steps = args.initial_random_steps
    policy_update_frequency = args.policy_update_frequency
    num_agents = args.num_agents

    # 2 agents
    trained_models_dir = './src/model_weights/msac/tworobot_weights.pth'
    agent = MSAC(state_dim, action_dim, replay_buff_size, batch_size, gamma, tau, initial_random_steps, policy_update_frequency, num_agents)
    agent.load_state_dict(torch.load(trained_models_dir))
    
    
    print('Testing mode')
    total_return = 0.
    total_step = 0
    total_path_len = 0.
    arrive_cnt = 0
    robot_name='turtlebot3_burger_1'

    while True:
        done = False
        score = [0 for _ in range(num_agents)]
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

            for i in range(num_agents):
                score[i] += r[i]

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

        for i in range(num_agents):
            print("score: ", score[i])

def plot(
    episode: int,
    scores: List[float],
    avg_scores: List[float],
    actor_losses: List[float],
    qf_losses: List[float],
    v_losses: List[float],
    alpha_losses: List[float]
    ):

    #plot dir
    plot_dir = './src/plots/'
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    if len(avg_scores) > 0:
        plt.title("Average reward per 100 episodes. Score: %s" % (avg_scores[-1]))
    else:
        plt.title("Average reward over 100 episodes.")
    plt.plot([100 + i for i in range(len(avg_scores))], avg_scores)
    plt.subplot(122)
    plt.title("episode %s. Score: %s" % (episode, np.mean(scores[-10:])))
    plt.plot(scores)
    

    plt.savefig(plot_dir + '/masac_result.png')
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.title('Actor Loss')
    plt.plot(actor_losses)
    plt.subplot(142)
    plt.title('qf loss')
    plt.plot(qf_losses)
    plt.subplot(143)
    plt.title('Vf Loss')
    plt.plot(v_losses)
    plt.subplot(144)
    plt.title('alpha loss')
    plt.plot(alpha_losses)
    plt.savefig(plot_dir + '/masac_loss.png')

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
    parser.add_argument('--policy-update-frequency', type=int, default=2, help='policy update frequency')

    #device
    parser.add_argument('--device', type=str, default='cpu', help='Which devices to use, cuda or cpu')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    args = parser.parse_args()

    train(args)
