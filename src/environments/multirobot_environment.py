#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState, ModelStates

from cv_bridge import CvBridge, CvBridgeError
from monodepth2 import *
import PIL
from matplotlib import cm
import torch
from pathlib import Path

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = './misc/gazebo_models/goal.sdf'
encoder, depth_decoder, w, h = get_depth_model("mono+stereo_640x192")
# Path('frames/rgb/').mkdir(parents=True, exist_ok=True)
# Path('frames/dis_map/').mkdir(parents=True, exist_ok=True)

class SingleRobot():
    def __init__(self, robot_name, model_name, goal_pos):
        self.position = Pose()
        self.original_position = Pose()

        self.goal_position = goal_pos
        self.robot_name = robot_name
        self.model_name = model_name
  
        self.past_distance = 0.
        self.pub_cmd_vel = rospy.Publisher(robot_name + '/cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber(robot_name + '/odom', Odometry, self.getOdometry)

        #  Original position for later respawn
        self.original_position = None

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        if self.original_position == None:
            self.original_position = self.position
        # get goal distance
        self.goal_distance = self.getGoalDistace()

        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

class Env():
    def __init__(self, is_training, train_env_id, test_env_id=2, num_agents=2, visual_obs=False, num_scan_ranges=10):
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id
        self.visual_obs = visual_obs
        self.num_agents = num_agents
        self.num_scan_ranges = num_scan_ranges
        self.n_step = 0

        #---------OBJECTS------------------#
        self.model_names = ["car_wheel", "cricket_ball", "beer", "cardboard_box", "first_2015_trash_can"]
        self.object_positions = []
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=5)
            except:
                pass
        for model_name in self.model_names:
            model_state = data.pose[data.name.index(model_name)].position
            self.object_positions.append((model_state.x, model_state.y))

        #--------------------------------#
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        

        if self.test_env_id == 4:
            self.test_goals = [(3.,3.), (-3.,2.), (3.,-3.), (-3., -1.2)]
        elif self.test_env_id == 1:
            # self.test_goals = [(2.3,-0.6), (3.5,3.), (1.5,5.2), (-2., 5.5)]
            self.test_goals = [(5.2,1), (5,3.), (1.5,5.2),(-2., 4.5)]
        elif self.test_env_id == 2:
            #self.test_goals = [(5.2,-0.3), (7,3.), (3, 5), (-1.,6.), (-3., 3.), (0.,0.)]
            self.test_goals = [(5.2,2), (5.2, 4), (3, 5), (-1.,5.), (-3., 3.), (0.,0.)]
        elif self.test_env_id == 3:
            self.test_goals = []
            for i in range(50):
                x, y = random.uniform(-2.4, 2.4), random.uniform(-2.4, 2.4)
                self.test_goals.append((x,y))
        else:
            print('No testing goal, let set it')
            exit(0)

        self.test_goals_id = 0
        self.is_training = is_training
        if self.is_training:
            self.threshold_arrive = 0.6
        else:
            self.threshold_arrive = 0.8

        #-----------GOAL-------------#
        self.goal_position = Pose()

        self.object_target_id = np.random.randint(len(self.model_names))
        self.goal_position.position.x = self.object_positions[self.object_target_id][0]
        self.goal_position.position.y = self.object_positions[self.object_target_id][1]

        #------------ROBOTS----------------#
        self.agents = []
        for i in range(self.num_agents):
            robot_name = "robot"+str(i+1)
            model_name = "turtlebot3_burger_"+str(i+1)
            self.agents.append(SingleRobot(robot_name, model_name, self.goal_position))
        
    
    def getState(self, scans, images):
        min_range = 0.13
        if self.is_training:
            min_range = 0.2
        
        obs_s, current_distance_s, yaw_s, rel_theta_s, diff_angle_s, die_s, arrive_s = [], [], [], [], [], [], []

        for i in range(self.num_agents):
            agent = self.agents[i]
            scan = scans[i]
            image = images[i]
            
            scan_range = []
            yaw = agent.yaw
            rel_theta = agent.rel_theta
            diff_angle = agent.diff_angle
            
            die = False
            arrive = False

            cof = (len(scan.ranges) / (self.num_scan_ranges - 1))
            for i in range(0, self.num_scan_ranges):
                n_i = math.ceil(i*cof - 1)
                if n_i < 0:
                    n_i = 0
                if cof == 1:
                    n_i = i
                if scan.ranges[n_i] == float('Inf'):
                    scan_range.append(3.5)
                elif np.isnan(scan.ranges[n_i]):
                    scan_range.append(0)
                else:
                    scan_range.append(scan.ranges[n_i])

            if min_range > min(scan_range) > 0: #collide
                current_distance = math.hypot(agent.goal_position.position.x - agent.position.x, agent.goal_position.position.y - agent.position.y)

                agent_object_dists = []
                for k, obj_name in enumerate(self.model_names):
                    agent_object_dists.append(math.hypot(self.object_positions[k][0] - agent.position.x, self.object_positions[k][1] - agent.position.y))

                index_min = agent_object_dists.index(min(agent_object_dists))
                if index_min == self.object_target_id:
                    print("Arrive!!!")
                    arrive = True

                die = True
                model_state = ModelState()
                model_state.model_name = agent.model_name
                # assign original position, or random
                model_state.pose.position.x = agent.original_position.x
                model_state.pose.position.y = agent.original_position.y
                model_state.pose.position.z = agent.original_position.z

                # Respawn the robot
                self.gazebo_model_state_service(model_state)

            # re-calculate the distance
            current_distance = math.hypot(agent.goal_position.position.x - agent.position.x, agent.goal_position.position.y - agent.position.y)

            obs = None
            if self.visual_obs:
                image = PIL.Image.fromarray(image)
                di = get_depth(image, encoder, depth_decoder, w, h)[0] # get disparity map
                data = di.squeeze(0).squeeze(0).cpu().numpy()
                rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                # im = PIL.Image.fromarray(rescaled)
                # im.save('frames/dis_map/' + str(self.n_step) + '.png')
                # image.save('frames/rgb/' + str(self.n_step) + '.png')
                spec_row = rescaled[rescaled.shape[0] // 2] # choose the middle row of the map

                # sample 10 value from this row as observation
                obs, n_samp = [], 10
                cof = len(spec_row)*1.0 / (n_samp - 1)
                for i in range(0, n_samp):
                    n_i = math.ceil(i*cof - 1)
                    if n_i < 0:
                        n_i = 0
                    if cof == 1:
                        n_i = i
                    obs.append(1. / (spec_row[n_i] + 0.00001))
            else:
                obs = scan_range

            obs_s.append(obs)
            current_distance_s.append(current_distance)
            yaw_s.append(yaw)
            rel_theta_s.append(rel_theta)
            diff_angle_s.append(diff_angle)
            die_s.append(die)
            arrive_s.append(arrive)

        return obs_s, current_distance_s, yaw_s, rel_theta_s, diff_angle_s, die_s, arrive_s
        
    def createGoal(self, is_reset):
        """
        is_reset: reset or is running in an episode
        """

        if self.is_training:
                self.object_target_id = np.random.randint(len(self.model_names))
                self.goal_position.position.x = self.object_positions[self.object_target_id][0]
                self.goal_position.position.y = self.object_positions[self.object_target_id][1]
                #update goal position
                for i in range(self.num_agents):
                    self.agents[i].goal_position = self.goal_position
                
        else:
            if is_reset:
                self.goal_position.position.x, self.goal_position.position.y = self.test_goals[self.test_goals_id]
            else:  
                self.test_goals_id += 1
                if self.test_goals_id >= len(self.test_goals):
                    pass
                    #print('FINISHED!!!')
                    #exit(0)
                else:
                    self.goal_position.position.x, self.goal_position.position.y = self.test_goals[self.test_goals_id]

        if is_reset:
            # build new target, only spawn in the beginning of the episode
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')

            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
        else:
            # only need to re-set target model state
            model_state = ModelState()
            model_state.model_name = 'target'
            # assign position
            model_state.pose.position.x = self.goal_position.position.x
            model_state.pose.position.y = self.goal_position.position.y
            model_state.pose.position.z = self.goal_position.position.z 
            # Respawn the target
            self.gazebo_model_state_service(model_state)

    def setEachReward(self, die_s, arrive_s):
        reward_s = []
        for i in range(self.num_agents):
            die = die_s[i]
            arrive = arrive_s[i]
            agent = self.agents[i]

            current_distance = math.hypot(agent.goal_position.position.x - agent.position.x, agent.goal_position.position.y - agent.position.y)
            distance_rate = (agent.past_distance - current_distance)

            reward = 500.*distance_rate
            agent.past_distance = current_distance

            if die:
                reward = -100.
                agent.pub_cmd_vel.publish(Twist())
                if not self.is_training and self.test_goals_id < len(self.test_goals):
                    self.test_goals_id += 1

            if arrive:
                reward = 120.
                agent.pub_cmd_vel.publish(Twist())
                
                agent.goal_distance = agent.getGoalDistace()
                arrive = False

            die_s[i] = die
            arrive_s[i] = arrive
            reward_s.append(reward)

        return reward_s

    def step(self, action_s, past_action_s):
        """
        action_n: 2 actions of n robots
        past_action_n: 2 past actions of n robots
        """
        data_s, image_s = [], []
        for i in range(self.num_agents):
            agent = self.agents[i]
            linear_vel = action_s[i][0]
            ang_vel = action_s[i][1]

            vel_cmd = Twist()
            vel_cmd.linear.x = linear_vel / 4
            vel_cmd.angular.z = ang_vel
            agent.pub_cmd_vel.publish(vel_cmd)
            
            robot_name = agent.robot_name

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message(robot_name + '/scan', LaserScan, timeout=5)
                except:
                    pass

            image = None
            while self.visual_obs == True and image is None:
                try:
                    image = rospy.wait_for_message(robot_name + '/camera1/image_raw', Image, timeout=5)
                    bridge = CvBridge()
                    try:
                        # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                        image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
                    except Exception as e:
                        raise e
                except:
                    pass
            
            data_s.append(data)
            image_s.append(image)


        state_s, rel_dis_s, yaw_s, rel_theta_s, diff_angle_s, die_s, arrive_s = self.getState(data_s, image_s)
        
        if self.visual_obs:     
            for i in range(len(state_s)):
                state_s[i] = [j / max(state_s[i]) for j in state_s[i]]
        else:
            for i in range(len(state_s)):
                state_s[i] = [j / 3.5 for j in state_s[i]]
                
        for i in range(self.num_agents):
            for pa in past_action_s[i]:
                state_s[i].append(pa)
        
        for i in range(self.num_agents):
            state_s[i] = state_s[i] + [rel_dis_s[i] / diagonal_dis, yaw_s[i] / 360, rel_theta_s[i] / 360, diff_angle_s[i] / 180]
            state_s[i] = np.array(state_s[i])
        

        r_s = self.setEachReward(die_s, arrive_s)
        self.n_step += 1

        # Reset goal if collided
        if sum(arrive_s) == 1:
            self.createGoal(is_reset=False)
        
        # return np.asarray(state), reward, done, arrive
        return state_s, r_s, die_s, arrive_s

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        # Create Goal
        self.createGoal(is_reset=True)

        # Get data
        data_s, image_s = [], []
        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.goal_distance = agent.getGoalDistace()
            robot_name = agent.robot_name

            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message(robot_name + '/scan', LaserScan, timeout=5)
                except:
                    pass

            image = None
            while self.visual_obs == True and image is None:
                try:
                    image = rospy.wait_for_message(robot_name + '/camera1/image_raw', Image, timeout=5)
                    bridge = CvBridge()
                    try:
                        # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                        image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
                    except Exception as e:
                        raise e
                except:
                    pass
            
            data_s.append(data)
            image_s.append(image)

            
        state_s, rel_dis_s, yaw_s, rel_theta_s, diff_angle_s, die_s, arrive_s = self.getState(data_s, image_s)

        if self.visual_obs:     
            for i in range(len(state_s)):
                state_s[i] = [j / max(state_s[i]) for j in state_s[i]]
        else:
            for i in range(len(state_s)):
                state_s[i] = [j / 3.5 for j in state_s[i]]
                
        for i in range(self.num_agents):
            state_s[i].append(0)
            state_s[i].append(0)

        for i in range(self.num_agents):
            state_s[i] = state_s[i] + [rel_dis_s[i] / diagonal_dis, yaw_s[i] / 360, rel_theta_s[i] / 360, diff_angle_s[i] / 180]
            state_s[i] = np.array(state_s[i])

        
        # return np.asarray(state)
        return state_s
