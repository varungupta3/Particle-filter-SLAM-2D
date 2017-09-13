from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from SE2 import SE2
import MapUtils as mp
from slam_utils_v2 import *
import time

def dead_reckoning(robot_pose, odo_curr, odo_prev):
	# Compute the change in odometry using Smart Subtraction 
	delta_odo_local = odo_curr - odo_prev
	# print robot_pose

	# Rotate the change in odometry to get change in global frame
	yaw_est = robot_pose[2]
	R = np.squeeze(np.array([[np.cos(yaw_est), -np.sin(yaw_est)], [np.sin(yaw_est), np.cos(yaw_est)]]))
	# print R
	particle_delta_pose_global = np.reshape(np.matmul(R, delta_odo_local.p),(2,1))
	# print particle_delta_pose_global

	# Update the robot's pose
	robot_pose[0:2] = robot_pose[0:2] + particle_delta_pose_global
	robot_pose[2] = robot_pose[2] + delta_odo_local.theta
	return robot_pose


def localization_prediction(num_particles, part_pose, odo_curr, odo_prev, W):
	
	# Randomly generate noise for all particles
	noise = np.random.multivariate_normal(np.array([0.,0.,0.]), W, num_particles)
	# print 'Max noise = ', np.max(noise, axis=0)

	# Compute the change in odometry using Smart Subtraction 
	delta_odo_local = odo_curr - odo_prev
	# print 'delta_odo_local = ', delta_odo_local.p, delta_odo_local.theta

	# Obtain all the rotation matrices and stack them in a list
	yaw_est = part_pose[:,2]
	R = [np.array([[np.cos(yaw_est[i]), -np.sin(yaw_est[i])], [np.sin(yaw_est[i]), np.cos(yaw_est[i])]]) for i in range(0, num_particles)]
	particle_delta_pose_global = np.zeros((num_particles, 2))

	# Rotate the change in pose from particle frame to get change in global frame
	for j in xrange(0, num_particles):
		particle_delta_pose_global[j,:] = np.matmul(R[j], delta_odo_local.p)
	# print 'delta pose = ', particle_delta_pose_global


	# Predict the particles pose using the odometry and noise
	part_pose[:,0:2] = part_pose[:,0:2] + particle_delta_pose_global + noise[:,0:2]
	part_pose[:,2] = part_pose[:,2] + delta_odo_local.theta + noise[:,2]
	# print 'part_pose = ', part_pose
	# time.sleep(5)
	return part_pose


def localization_update(num_particles, particle_pose, particle_wts, z_t, MAP, LIDAR, ROBOT, N_thresh, h_angle, n_angle):

	map_corr = np.zeros((num_particles, 1))
	grid_size = 1 # Odd number. Even number becomes the next biggest odd number

	# Decide the grid size
	xs = np.arange(-np.int_(grid_size/2)*MAP['x_res'], np.int_((grid_size)/2)*MAP['x_res'] + 0.01, MAP['x_res'])
	ys = np.arange(-np.int_(grid_size/2)*MAP['y_res'], np.int_((grid_size)/2)*MAP['y_res'] + 0.01, MAP['y_res'])
	# print 'xs = ', xs, 'ys = ', ys

	# Create a boolean version of the map
	bool_map = np.copy(MAP['map'])
	bool_map[bool_map>0] = 1
	bool_map[bool_map<=0] = 0
	# print bool_map
	x, y, z = get_body_frame_coordinates(z_t, LIDAR, ROBOT, h_angle, n_angle) # Get coordinates in body frame
	num_valid_scans = x.size
	for j in xrange(0, num_particles):
		# Convert to global frame
		yaw_est = particle_pose[j,2]
		R = np.array([[np.cos(yaw_est), -np.sin(yaw_est), particle_pose[j,0]], [np.sin(yaw_est), np.cos(yaw_est), particle_pose[j,1]]])
		scan_global = np.matmul(R, np.vstack((x, y, np.ones((num_valid_scans)))))
		# Calculate correlation with previous map
		corr = mp.mapCorrelation(bool_map, MAP['x_idx_all'], MAP['y_idx_all'], scan_global, xs, ys)
		map_corr[j,0] = np.max(corr)
	# print map_corr
	print 'Max Correlation : ', np.max(map_corr)
	# logwts = np.log(particle_wts) + map_corr
	# logsumexp = np.log(np.sum(np.exp(logwts-np.max(logwts))))
	# particle_wts = np.exp(logwts - np.max(logwts) - logsumexp)

	particle_wts *= map_corr
	sum_wts = np.sum(particle_wts)
	particle_wts  /= sum_wts

	# print 'Particle weights : ', particle_wts
	
	best_idx = np.argmax(particle_wts)
	robot_pose = particle_pose[best_idx, :]

	N_eff = 1./np.sum(np.power(particle_wts, 2))
	print 'Effective particles = ', N_eff
	if N_eff < N_thresh:
		ind = resampleWeights(particle_wts)
		# print np.unique(ind), np.unique(ind).size
		# print 'Resampled Indices = ', ind
		particle_pose = np.squeeze(particle_pose[ind,:])
		# print 'Resampled Particle Pose = ', particle_pose
		particle_wts = np.ones((num_particles, 1)) * 1./num_particles
	return robot_pose, particle_pose, particle_wts


def mapping(scan, MAP, LIDAR, JOINT, ROBOT, h_angle, n_angle, max_times):
	cmap = np.copy(MAP['map'])
	# tidx = np.argmin(np.abs(LIDAR['t'][lidar_idx] - JOINT['t']))
	# h_angle = JOINT['head_angle'][tidx]
	# n_angle = JOINT['neck_angle'][tidx]
	# print 'Joint index : ', tidx, 'Head Angle : ', h_angle, 'Neck Angle : ', n_angle
	x, y, z = get_body_frame_coordinates(scan, LIDAR, ROBOT, h_angle, n_angle)
	num_valid_scans = x.size
	yaw_est = ROBOT['pose'][2,0]
	R = np.array([[np.cos(yaw_est), -np.sin(yaw_est), ROBOT['pose'][0,0]], [np.sin(yaw_est), np.cos(yaw_est), ROBOT['pose'][1,0]]])
	scan_global = np.matmul(R, np.vstack((x, y, np.ones(num_valid_scans))))
	x_idx, y_idx = get_indices(scan_global[0,:], scan_global[1,:], MAP)
	cmap[x_idx, y_idx] += 1.25 * MAP['LOG_ODDS']

	l2h_origin = lidar2head(np.array([[0.],[0.],[0.],[1.]]), h_angle, n_angle, ROBOT)
	global_origin = np.matmul(np.array([[np.cos(yaw_est), -np.sin(yaw_est)], [np.sin(yaw_est), np.cos(yaw_est)]]), np.vstack((ROBOT['pose'][0,0], ROBOT['pose'][1,0])))
	robot_x_idx, robot_y_idx = get_indices(ROBOT['pose'][0,0] + global_origin[0,0], ROBOT['pose'][1,0] + global_origin[1,0], MAP)
	occupied_indices = np.int_(np.hstack((np.vstack((y_idx, x_idx)),np.array([[robot_y_idx],[robot_x_idx]]))))
	mask = np.zeros(cmap.shape)
	cv2.drawContours(image=mask, contours=[occupied_indices.T], contourIdx=0, color=-0.25*MAP['LOG_ODDS'], thickness=-1)
	cmap += mask
	cmap[cmap > max_times*MAP['LOG_ODDS']] = max_times*MAP['LOG_ODDS']
	cmap[cmap < -max_times*MAP['LOG_ODDS']] = -max_times*MAP['LOG_ODDS']
	return cmap, occupied_indices[:,0:-1]


