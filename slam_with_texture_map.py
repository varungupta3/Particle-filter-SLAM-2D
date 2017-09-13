from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, os.path
from scipy import io
from load_data import *
import cv2
from SE2 import SE2
import time
import sys
import MapUtils as mp
import pdb
import cProfile, pstats, StringIO
from slam_utils_v2 import *
from slam_methods import *
from mpl_toolkits.mplot3d import Axes3D
import pickle

def load_data():
    # lidar = get_lidar('C:/Users/Varun/Dropbox/Penn_Courses/ESE 650 Learning in Robotics/Project 3/Codes/data/train_lidar2')
    # joint = get_joint('C:/Users/Varun/Dropbox/Penn_Courses/ESE 650 Learning in Robotics/Project 3/Codes/data/train_joint2')
    lidar = get_lidar('./data/train_lidar0')
    joint = get_joint('./data/train_joint0')
    # lidar = get_lidar('../Proj3_2017_test/test_lidar')
    # joint = get_joint('../Proj3_2017_test/test_joint')

    # Readings from LIDAR
    LIDAR = {}
    LIDAR['scan'] = np.squeeze([lidar[i]['scan'] for i in range(0, len(lidar))])
    LIDAR['pose'] = np.squeeze([lidar[i]['pose'] for i in range(0, len(lidar))])
    LIDAR['t'] = np.squeeze([lidar[i]['t'] for i in range(0, len(lidar))])
    LIDAR['res'] = np.squeeze(lidar[0]['res'])
    LIDAR['theta'] = np.arange(-135 * np.pi/180, 135 * np.pi/180 + LIDAR['res'], LIDAR['res'])[0:-1]
    LIDAR['maxrange'] = 30.0
    LIDAR['minrange'] = 0.1

    # Readings from JOINT DATA
    JOINT = {}
    JOINT['head_angle'] = joint['head_angles'][1,:]
    JOINT['neck_angle'] = joint['head_angles'][0,:]
    JOINT['t'] = np.squeeze(joint['ts'])

    # Robot Parameters
    ROBOT = {}
    ROBOT['h_CM'] = 0.93
    ROBOT['h_HEAD'] = ROBOT['h_CM'] + 0.33
    ROBOT['h_LIDAR'] = ROBOT['h_HEAD'] + 0.15
    ROBOT['pose'] = np.reshape(np.array([0.,0.,0.]),(3, 1))

    return LIDAR, JOINT, ROBOT


def initMap():
    MAP = {}
    MAP['x_dist'] = 40.0
    MAP['x_res'] = 0.1
    MAP['y_dist'] = 40.0
    MAP['y_res'] = 0.1
    MAP['grid_rows'] = np.int(2 * np.floor(MAP['y_dist']/MAP['y_res']) + 1)
    MAP['grid_cols'] = np.int(2 * np.floor(MAP['x_dist']/MAP['x_res']) + 1)
    MAP['map'] = np.zeros((MAP['grid_rows'], MAP['grid_cols']))
    MAP['x_idx_all'] = np.arange(-MAP['x_dist'], MAP['x_dist'] + MAP['x_res'], MAP['x_res'])
    MAP['y_idx_all'] = np.arange(-MAP['y_dist'], MAP['y_dist'] + MAP['y_res'], MAP['y_res'])
    MAP['LOG_ODDS'] = np.log(9.)
    return MAP


def initPF():
    PF = {}
    PF['num_particles'] = 100
    PF['N_thresh'] = 0.9 * PF['num_particles']
    PF['particle_pose'] = np.tile(np.array([0., 0., 0.]), (PF['num_particles'], 1))
    PF['particle_wts'] = np.ones((PF['num_particles'], 1)) * 1./PF['num_particles']
    PF['W'] = np.diag([0.001, 0.001, 0.001])
    return PF


def run_slam():

    videoWrite = False
    plotRealTime = True
    createTextureMap = True

    frameNum = 0

    pr = cProfile.Profile()
    pr.enable()

    # Load Lidar, Joint data and robot parameters.
    LIDAR, JOINT, ROBOT = load_data()

    # Initialize map and particles
    MAP = initMap()
    PF = initPF()

    if createTextureMap:
        depth = io.loadmat('../Proj3_2017_Train_rgb/DEPTH_0.mat')
        rgb_images = io.loadmat('../Proj3_2017_Train_rgb/RGB_0.mat')
        depth = depth['DEPTH'][0]
        rgb_images = rgb_images['RGB'][0]
        num_rgb = len(rgb_images)
        num_depth = len(depth)
        t_depth = np.zeros(num_depth)
        img_index = np.zeros(num_depth)
        for didx in range(num_depth):
            t_depth[didx] = depth[didx][0][0][0]
            img_index[didx] = np.argmin(np.abs(LIDAR['t'] - t_depth[didx]))
        img_index = np.round(img_index/10.)*10
        # rgb = pickle.load(open('RGBcamera_Calib_result.pkl','rb'))
        # fc_rgb = rgb['fc']
        fc_rgb = [1049.331752604831308, 1051.318476285322504]
        fc_depth = [364.457362485643273, 364.542810626989194]
        MAP['tmap'] = np.zeros(MAP['map'].shape)
        # images = []
        # for rgbidx in range(num_rgb):
        #     images = rgb_images[0][0][0][-1]
        # im_size = images.shape

    print 'Initializations Done!'

    # Remove the irrelevant data in the start where the robot doesn't move
    # start_idx = get_start_idx(LIDAR)
    start_idx = 10

    print 'start_idx = ', start_idx

    # Update the map using the first scan
    MAP['map'], scanned_indices = get_first_scan(MAP, LIDAR, JOINT, ROBOT, start_idx)

    # print 'MAP = ', MAP['map']

    # Initialize the robots location in the occupancy grid
    robot_pose = np.vstack((ROBOT['pose']))
    # print robot_pose
    robot_pose_x_idx, robot_pose_y_idx = get_indices(ROBOT['pose'][0,0], ROBOT['pose'][1,0], MAP)
    # print robot_pose_x_idx, robot_pose_y_idx
    robot_pose_idx = np.vstack((robot_pose_y_idx, robot_pose_x_idx))

    # print 'Robot Pose index = ', robot_pose_idx

    # Number of scans to jump for creating the map
    jump_scans = 10

    # Max times the log odds are increased
    max_times = 1000

    # Obtain initial odometry (relative pose)
    odo_prev = SE2(LIDAR['pose'][start_idx-jump_scans, 0:2], LIDAR['pose'][start_idx-jump_scans, 2])

    plt.ion()

    if videoWrite:
        videoWriteParams = {'folderName':'./output/testmap2/', 'writeVideo':videoWrite}
        plot_map(MAP, robot_pose, 'GRAYSCALE', scanned_indices, videoWriteParams, frameNum)
    else:
        plot_map(MAP, robot_pose, 'GRAYSCALE', scanned_indices)
        # plot_map(MAP, robot_pose, 'COLORED', scanned_indices)
    print 'First Scan Plotted!'
    time = 0

    # if createTextureMap:
    #     files = 1

    for i in xrange(start_idx, len(LIDAR['t']), jump_scans):
        print '-----------------------------------------------------------------------------------'
        print 'Scan Number : ', i
        time = np.append(time, LIDAR['t'][i]-LIDAR['t'][start_idx])
        frameNum += 1

        # if createTextureMap:
        #     rgbidx = np.argmin(np.abs(LIDAR['t'][i] - t_depth))
        #     image = rgb_images[rgbidx][0][0][-1]

        # Obtain the closest possible head and neck angles
        idx = np.argmin(np.abs(LIDAR['t'][i] - JOINT['t']))
        h_angle = JOINT['head_angle'][idx]
        n_angle = JOINT['neck_angle'][idx]

        # Read the scanned distances at instant t
        z_t = LIDAR['scan'][i]
        # x, y, z = get_global_coordinates(z_t, LIDAR, h_angle, n_angle)
        odo_curr = SE2(LIDAR['pose'][i,0:2], LIDAR['pose'][i,2])

        PF['particle_pose'] = localization_prediction(PF['num_particles'], PF['particle_pose'], odo_curr, odo_prev, PF['W'])
        true_pose_est, PF['particle_pose'], PF['particle_wts'] = localization_update(PF['num_particles'], PF['particle_pose'], PF['particle_wts'], z_t, MAP, LIDAR, ROBOT, PF['N_thresh'], h_angle, n_angle)

        x_pos_idx, y_pos_idx = get_indices(true_pose_est[0], true_pose_est[1], MAP)
        robot_pose_idx = np.hstack((robot_pose_idx, np.reshape(np.array([y_pos_idx, x_pos_idx]), (2,1))))
        odo_prev = odo_curr

        ROBOT['pose'] = np.reshape(np.copy(true_pose_est), (3,1))
        robot_pose = np.hstack((robot_pose, np.reshape(true_pose_est, (3,1))))

        MAP['map'], scanned_indices = mapping(z_t, MAP, LIDAR, JOINT, ROBOT, h_angle, n_angle, max_times)

        if createTextureMap:
            if np.any(img_index == i):
                img_no = np.where(img_index == i)
                MAP['tmap'] = texture_map(MAP, depth[img_no][0][0][-1], rgb_images[img_no][0][0][-1],
                                   robot_pose[:, -1], fc_rgb, fc_depth, h_angle, n_angle)


        if plotRealTime:
            if np.mod(i-start_idx, 200) == 0:
                if videoWrite:
                    plot_map(MAP, robot_pose, 'GRAYSCALE', scanned_indices, videoWriteParams, frameNum)
                else:
                    plot_map(MAP, robot_pose, 'GRAYSCALE', scanned_indices)
                    # plot_map(MAP, robot_pose, 'COLORED', scanned_indices)


    plt.subplot(3, 1, 1)
    plt.plot(time, robot_pose[0,:], 'ko-')
    plt.xlabel('time (s)')
    plt.ylabel('X')

    plt.subplot(3, 1, 2)
    plt.plot(time, robot_pose[1,:], 'r.-')
    plt.xlabel('time (s)')
    plt.ylabel('Y')

    plt.subplot(3, 1, 3)
    plt.plot(time, robot_pose[2,:], 'r.-')
    plt.xlabel('time (s)')
    plt.ylabel('theta (rad)')

    plt.show()
    plt.pause(10)

    plot_map(MAP, robot_pose, 'GRAYSCALE')
    # plot_map(MAP, robot_pose, 'COLORED')
    plt.pause(100)

    pr.disable()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xRange = np.arange(-MAP['x_dist'], MAP['x_dist'] + MAP['x_res'], MAP['x_res'])
    yRange = np.arange(-MAP['y_dist'], MAP['y_dist'] + MAP['y_res'], MAP['y_res'])
    X, Y = np.meshgrid(xRange, yRange)
    # print X, Y
    surf = ax.plot_surface(X, Y, MAP['map'], cmap='winter', linewidth=0, antialiased=False)
    plt.show()

    # pdb.set_trace()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()


if __name__ == '__main__':
    run_slam()
