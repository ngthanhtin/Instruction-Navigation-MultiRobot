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
from gazebo_msgs.msg import ModelState

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

class Env():
    def __init__(self, is_training, train_env_id, test_env_id=2, visual_obs=False, num_scan_ranges=10):
        self.train_env_id = train_env_id
        self.test_env_id = test_env_id
        self.visual_obs = visual_obs
        self.num_scan_ranges = num_scan_ranges
        self.n_step = 0

        #-----------GOAL-------------#
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.

        #------------ROBOT 1----------------#
        self.position_1 = Pose()
        self.past_distance_1 = 0.
        self.pub_cmd_vel_1 = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        self.sub_odom_1 = rospy.Subscriber('/robot1/odom', Odometry, self.getOdometry, 1)

        #------------ROBOT 2----------------#
        self.position_2 = Pose()
        self.past_distance_2 = 0.
        self.pub_cmd_vel_2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=10)
        self.sub_odom_2 = rospy.Subscriber('/robot2/odom', Odometry, self.getOdometry, 2)

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
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def getGoalDistace(self, index):
        if index == 1:
            goal_distance = math.hypot(self.goal_position.position.x - self.position_1.x, self.goal_position.position.y - self.position_1.y)
            self.past_distance_1 = goal_distance
            return goal_distance
        elif index == 2:
            goal_distance = math.hypot(self.goal_position.position.x - self.position_2.x, self.goal_position.position.y - self.position_2.y)
            self.past_distance_2 = goal_distance
            return goal_distance
        else:
            print("Out of index.....")

        

    def getOdometry(self, odom, index):
        if index == 1:
            self.position_1 = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
            yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

            if yaw >= 0:
                yaw = yaw
            else:
                yaw = yaw + 360

            rel_dis_x = round(self.goal_position.position.x - self.position_1.x, 1)
            rel_dis_y = round(self.goal_position.position.y - self.position_1.y, 1)

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

            self.rel_theta_1 = rel_theta
            self.yaw_1 = yaw
            self.diff_angle_1 = diff_angle

        elif index == 2:
            self.position_2 = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
            yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

            if yaw >= 0:
                yaw = yaw
            else:
                yaw = yaw + 360

            rel_dis_x = round(self.goal_position.position.x - self.position_2.x, 1)
            rel_dis_y = round(self.goal_position.position.y - self.position_2.y, 1)

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

            self.rel_theta_2 = rel_theta
            self.yaw_2 = yaw
            self.diff_angle_2 = diff_angle

        else:
            print("Out of index...")

    def getState(self, scan, image, index):
        if index == 1:
            scan_range = []
            yaw = self.yaw_1
            rel_theta = self.rel_theta_1
            diff_angle = self.diff_angle_1
            min_range = 0.13
            if self.is_training:
                min_range = 0.2
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
                die = True
                model_state = ModelState()
                model_state.model_name = 'turtlebot3_burger_1'
                model_state.pose.position.x = 1
                model_state.pose.position.y = 1
                model_state.pose.position.z = 0

                # Respawn the robot
                self.gazebo_model_state_service(model_state)
                

            current_distance = math.hypot(self.goal_position.position.x - self.position_1.x, self.goal_position.position.y - self.position_1.y)
            if current_distance <= self.threshold_arrive:
                # done = True
                arrive = True

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
        
        elif index == 2:
            scan_range = []
            yaw = self.yaw_2
            rel_theta = self.rel_theta_2
            diff_angle = self.diff_angle_2
            min_range = 0.13
            if self.is_training:
                min_range = 0.2
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
                die = True
                model_state = ModelState()
                model_state.model_name = 'turtlebot3_burger_2'
                model_state.pose.position.x = -1
                model_state.pose.position.y = 1
                model_state.pose.position.z = 0

                # Respawn the robot
                self.gazebo_model_state_service(model_state)

            current_distance = math.hypot(self.goal_position.position.x - self.position_2.x, self.goal_position.position.y - self.position_2.y)
            if current_distance <= self.threshold_arrive:
                # done = True
                arrive = True

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

        else:
            print('Out of index...')

        return obs, current_distance, yaw, rel_theta, diff_angle, die, arrive

    def setReward(self, arrive_n, reward_n):
        """
        arrive_n: an array contains arrival of n robots
        reward_n : reward of n robots
        """
        if arrive_n[0] == True or arrive_n[1] == True: 
            # build new target
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                if self.is_training:
                    if self.train_env_id == 3:
                        while True:
                            x, y = random.uniform(-3.3, 3.3), random.uniform(-3.3, 3.3)
                            if 1.5 > abs(x) > 0.5 and abs(y) < 2.5:
                                continue
                            elif 2.5 > abs(y) > 2. and 0. > x > 1.5:
                                continue
                            else:
                                break
                        self.goal_position.position.x = x
                        self.goal_position.position.y = y
                    else:
                        while True:
                            x, y = random.uniform(-3.2, 3.2), random.uniform(-3.2, 3.2)
                            if abs(x) > 1. or abs(y) > 1.:
                                break
                        self.goal_position.position.x = x
                        self.goal_position.position.y = y
                else:
                    self.test_goals_id += 1
                    if self.test_goals_id >= len(self.test_goals):
                        pass
                        #print('FINISHED!!!')
                        #exit(0)
                    else:
                        self.goal_position.position.x, self.goal_position.position.y = self.test_goals[self.test_goals_id]

                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')

        return sum(reward_n)

    def setRewardIndex(self, die, arrive, index):
        if index == 1:
            current_distance = math.hypot(self.goal_position.position.x - self.position_1.x, self.goal_position.position.y - self.position_1.y)
            distance_rate = (self.past_distance_1 - current_distance)

            reward = 500.*distance_rate
            self.past_distance_1 = current_distance

            if die:
                reward = -100.
                self.pub_cmd_vel_1.publish(Twist())
                if not self.is_training and self.test_goals_id < len(self.test_goals):
                    self.test_goals_id += 1

            if arrive:
                reward = 120.
                self.pub_cmd_vel_1.publish(Twist())
                
                self.goal_distance_1 = self.getGoalDistace(index=1)
                arrive = False

        elif index == 2:
            current_distance = math.hypot(self.goal_position.position.x - self.position_2.x, self.goal_position.position.y - self.position_2.y)
            distance_rate = (self.past_distance_2 - current_distance)

            reward = 500.*distance_rate
            self.past_distance_2 = current_distance

            if die:
                reward = -100.
                self.pub_cmd_vel_2.publish(Twist())
                if not self.is_training and self.test_goals_id < len(self.test_goals):
                    self.test_goals_id += 1

            if arrive:
                reward = 120.
                self.pub_cmd_vel_2.publish(Twist())
        
                self.goal_distance_2 = self.getGoalDistace(index=2)
                arrive = False
        else:
            print("Out of index...")
        return reward

    def step(self, action_n, past_action_n):
        """
        action_n: n actions of n robots
        past_action_n: n past actions of n robots
        """
        linear_vel_1 = action_n[0][0]
        ang_vel_1 = action_n[0][1]

        vel_cmd_1 = Twist()
        vel_cmd_1.linear.x = linear_vel_1 / 4
        vel_cmd_1.angular.z = ang_vel_1
        self.pub_cmd_vel_1.publish(vel_cmd_1)

        linear_vel_2 = action_n[1][0]
        ang_vel_2 = action_n[1][1]
        
        vel_cmd_2 = Twist()
        vel_cmd_2.linear.x = linear_vel_2 / 4
        vel_cmd_2.angular.z = ang_vel_2
        self.pub_cmd_vel_2.publish(vel_cmd_2)

        data = [None, None]
        while data[0] is None or data[1] is None:
            try:
                data[0] = rospy.wait_for_message('robot1/scan', LaserScan, timeout=5)
                data[1] = rospy.wait_for_message('robot2/scan', LaserScan, timeout=5)
            except:
                pass

        image = [None, None]
        while self.visual_obs == True and (image[0] is None or image[1] is None):
            try:
                image[0] = rospy.wait_for_message('/robot1/camera1/image_raw', Image, timeout=5)
                image[1] = rospy.wait_for_message('/robot2/camera1/image_raw', Image, timeout=5)
                bridge = CvBridge()
                try:
                    # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                    image[0] = bridge.imgmsg_to_cv2(image[0], desired_encoding="passthrough")
                    image[1] = bridge.imgmsg_to_cv2(image[1], desired_encoding="passthrough")
                except Exception as e:
                    raise e
            except:
                pass

        state_1, rel_dis_1, yaw_1, rel_theta_1, diff_angle_1, die_1, arrive_1 = self.getState(data[0], image[0], index=1)
        state_2, rel_dis_2, yaw_2, rel_theta_2, diff_angle_2, die_2, arrive_2 = self.getState(data[1], image[1], index=2)
        if self.visual_obs:
            state_1 = [i / max(state_1) for i in state_1]
            state_2 = [i / max(state_2) for i in state_2]
        else:
            state_1 = [i / 3.5 for i in state_1]
            state_2 = [i / 3.5 for i in state_2]

        for pa in past_action_n[0]:
            state_1.append(pa)
        
        for pa in past_action_n[1]:
            state_2.append(pa)

        state_1 = state_1 + [rel_dis_1 / diagonal_dis, yaw_1 / 360, rel_theta_1 / 360, diff_angle_1 / 180]
        state_1 = np.array(state_1)
        state_2 = state_2 + [rel_dis_2 / diagonal_dis, yaw_2 / 360, rel_theta_2 / 360, diff_angle_2 / 180]
        state_2 = np.array(state_2)

        state = [state_1, state_2]

        r1 = self.setRewardIndex(die_1, arrive_1, index=1)
        r2 = self.setRewardIndex(die_2, arrive_2, index=2)

        reward = self.setReward([arrive_1, arrive_2], [r1, r2])
        self.n_step += 1

        # done, arrive = False, False
        # if die_1 == True and die_2 == True:
        #     done = True
        
        # if arrive_1 == True and arrive_2 == True:
        #     arrive = True

        # return np.asarray(state), reward, done, arrive
        return state, [r1, r2], [die_1, die_2], [arrive_1, arrive_2]

    def reset(self):
        # Reset the env #
        
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            if self.is_training:
                if self.train_env_id == 3:
                    while True:
                        x, y = random.uniform(-3.3, 3.3), random.uniform(-3.3, 3.3)
                        if 1.5 > abs(x) > 0.5 and abs(y) < 2.5:
                            continue
                        elif 2.5 > abs(y) > 2. and 0. > x > 1.5:
                            continue
                        else:
                            break
                    self.goal_position.position.x = x
                    self.goal_position.position.y = y
                else:
                    while True:
                        x, y = random.uniform(-3.2, 3.2), random.uniform(-3.2, 3.2)
                        if abs(x) > 1. or abs(y) > 1.:
                            break
                    self.goal_position.position.x = x
                    self.goal_position.position.y = y
            else:
                self.goal_position.position.x, self.goal_position.position.y = self.test_goals[self.test_goals_id]

            # if -0.3 < self.goal_position.position.x < 0.3 and -0.3 < self.goal_position.position.y < 0.3:
            #     self.goal_position.position.x += 1
            #     self.goal_position.position.y += 1

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')

        data = [None, None]
        while data[0] is None or data[1] is None:
            try:
                data[0] = rospy.wait_for_message('robot1/scan', LaserScan, timeout=5)
                data[1] = rospy.wait_for_message('robot2/scan', LaserScan, timeout=5)
                
            except:
                pass

        image = [None, None]
        
        while self.visual_obs == True and (image[0] is None or image[1] is None):
            try:
                image[0] = rospy.wait_for_message('/robot1/camera1/image_raw', Image, timeout=5)
                image[1] = rospy.wait_for_message('/robot2/camera1/image_raw', Image, timeout=5)
                bridge = CvBridge()
                try:
                    # self.camera_image is an ndarray with shape (h, w, c) -> (228, 304, 3)
                    image[0] = bridge.imgmsg_to_cv2(image[0], desired_encoding="passthrough")
                    image[1] = bridge.imgmsg_to_cv2(image[1], desired_encoding="passthrough")
                    # frame = bridge.imgmsg_to_cv(image, "bgr8")
                    
                except Exception as e:
                    raise e
            except:
                pass

        self.goal_distance_1 = self.getGoalDistace(index=1)
        self.goal_distance_2 = self.getGoalDistace(index=2)

        state_1, rel_dis_1, yaw_1, rel_theta_1, diff_angle_1, die_1, arrive_1 = self.getState(data[0], image[0], index=1)
        state_2, rel_dis_2, yaw_2, rel_theta_2, diff_angle_2, die_2, arrive_2 = self.getState(data[1], image[1], index=2)
        if self.visual_obs:
            state_1 = [i / max(state_1) for i in state_1]
            state_2 = [i / max(state_2) for i in state_2]
        else:
            state_1 = [i / 3.5 for i in state_1]
            state_2 = [i / 3.5 for i in state_2]

        state_1.append(0)
        state_1.append(0)
        state_2.append(0)
        state_2.append(0)

        state_1 = state_1 + [rel_dis_1 / diagonal_dis, yaw_1 / 360, rel_theta_1 / 360, diff_angle_1 / 180]
        state_1 = np.array(state_1)
        state_2 = state_2 + [rel_dis_2 / diagonal_dis, yaw_2 / 360, rel_theta_2 / 360, diff_angle_2 / 180]
        state_2 = np.array(state_2)

        state = [state_1, state_2]
        # return np.asarray(state)
        return state
