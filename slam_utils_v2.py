from __future__ import division
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import matplotlib.animation as manimation
import cv2
import math
# from SE2 import SE2

def lidar2head(X, h_angle, n_angle, ROBOT):
    Rz = np.array([[np.cos(n_angle), -np.sin(n_angle), 0., 0.], [np.sin(n_angle), np.cos(n_angle), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) # Rotate by neck angle
    Ry = np.array([[np.cos(h_angle), 0., np.sin(h_angle), 0.], [0., 1., 0., 0.], [-np.sin(h_angle), 0., np.cos(h_angle), 0.], [0. ,0., 0., 1.]]) # Rotate by head angle
    T = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., ROBOT['h_LIDAR']-ROBOT['h_HEAD']], [0., 0., 0., 1.]])
    R = np.matmul(Rz, Ry)
    netT = np.matmul(R, T)
    Y = np.matmul(netT, X)
    return Y


def get_coordinates_in_lidar_frame(scan, theta):
    x = scan * np.cos(theta)
    y = scan * np.sin(theta)
    z = np.zeros((theta.size))
    return np.vstack((x,y,z,np.ones((theta.size))))


def check_valid_scan(scan, MIN_RANGE, MAX_RANGE):
    return scan * ((scan > MIN_RANGE) & (scan < MAX_RANGE))


def check_ground(z, scan):
    return (z > 0.1) & (scan != 0)


def get_body_frame_coordinates(scan, LIDAR, ROBOT, h_angle, n_angle):
    scan = check_valid_scan(scan, LIDAR['minrange'], LIDAR['maxrange']) # Check validity based on min and max range
    X = get_coordinates_in_lidar_frame(scan, LIDAR['theta'])
    Y = lidar2head(X, h_angle, n_angle, ROBOT)
    x = Y[0,:]
    y = Y[1,:]
    z = Y[2,:] + ROBOT['h_HEAD']
    # print 'Valid scans = ', x[np.nonzero(x)].size
    ground_check = check_ground(z, scan) # Check if hitting floor
    x = x * ground_check
    y = y * ground_check
    z = z * ground_check
    x = x[np.nonzero(x)]
    y = y[np.nonzero(y)]
    z = z[np.nonzero(z)]
    # print 'Acceptable scans = ', x.size
    return x, y, z


def get_indices(x, y, MAP):
    x_idx = np.int_(np.round((x+MAP['x_dist'])/MAP['x_res']))
    y_idx = np.int_(np.round((y+MAP['y_dist'])/MAP['y_res']))
    return x_idx, y_idx


def get_first_scan(MAP, LIDAR, JOINT, ROBOT, start_idx):
    map = np.copy(MAP['map'])

    # Identify the index of the starting point
    x_start_idx, y_start_idx = get_indices(ROBOT['pose'][0,0], ROBOT['pose'][1,0], MAP)
    robot_pose_idx = np.reshape(np.array([y_start_idx, x_start_idx]),(2, 1))

    # Obtain the head and neck angle from closest time stamp
    idx = np.argmin(np.abs(LIDAR['t'][start_idx] - JOINT['t']))
    h_angle = JOINT['head_angle'][idx]
    n_angle = JOINT['neck_angle'][idx]
    # print 'Joint index : ', idx, 'Head Angle : ', h_angle, 'Neck Angle : ', n_angle

    # Read first scan
    z_0 = LIDAR['scan'][start_idx]

    # Convert scan to global coordinates and eliminate invalid scans
    x, y, z = get_body_frame_coordinates(z_0, LIDAR, ROBOT, h_angle, n_angle)

    # Obtain the scanned points in the occupancy grid
    x_idx, y_idx = get_indices(x, y, MAP)

    # Update the map
    map[x_idx, y_idx] += 2 * MAP['LOG_ODDS']
    ### Try incresing log odds of repeated points by a bigger margin

    # Use the scanned grid points and robot position in the grid to obtain contours and update free region in the map
    # occupied_indices = np.int_(np.hstack((np.vstack((y_idx, x_idx)),np.array([[np.round(MAP['grid_cols']/2)],[np.round(MAP['grid_rows']/2)]]))))
    occupied_indices = np.int_(np.hstack((np.vstack((y_idx, x_idx)),np.array([[y_start_idx],[x_start_idx]]))))
    mask = np.zeros(map.shape)
    cv2.drawContours(image=mask, contours=[occupied_indices.T], contourIdx=0, color=-MAP['LOG_ODDS'], thickness=-1)
    map += mask
    return map, occupied_indices[:,0:-1]


def get_start_idx(LIDAR):
    eps1 = 0.00001
    eps2 = 0.00001
    for temp in xrange(1, len(LIDAR['t'])):
        deltaP = LIDAR['pose'][temp,:] - LIDAR['pose'][temp-1,:]
        if ((deltaP[0] > eps1) & (deltaP[1] > eps1)) | (deltaP[2] > eps2):
            break
    if temp > 100:
        start_idx = temp - 100
    else:
        start_idx = temp - 1
    return start_idx
    # print 'Start Index = ', start_idx


def plot_map(MAP, robot_pose, MAP_TYPE, scanned_indices = np.empty((0,0)), videoWriteParams = None, frameNum = 0):
    plt.clf()
    if MAP_TYPE == 'GRAYSCALE':
        map2 = np.copy(MAP['map'])
        map2[map2==0] = 0.5
        map2[((map2!=0.5) & (map2>-20*MAP['LOG_ODDS']))] = 0
        map2[map2<0] = 1
        plt.imshow(map2, cmap='gray')
    else:
        plt.imshow(MAP['map'])

    robot_pose = np.divide(robot_pose.T, np.array([MAP['x_res'], MAP['y_res'], 1])) + np.array([(MAP['grid_rows']-1)/2, (MAP['grid_cols']-1)/2, 0])
    robot_pose = robot_pose.T
    plt.plot(robot_pose[1,:], robot_pose[0,:], c='r') # Inverted coordinates (y, x)
    direction = robot_pose[2,-1]
    xline = [robot_pose[0,-1], robot_pose[0,-1] + (1./MAP['x_res'])*np.cos(direction)]
    yline = [robot_pose[1,-1], robot_pose[1,-1] + (1./MAP['y_res'])*np.sin(direction)]
    plt.plot(yline, xline, c='b')
    if scanned_indices.size:
        plt.plot(scanned_indices[0,:], scanned_indices[1,:], c='y')

    if videoWriteParams is not None:
        if videoWriteParams['writeVideo']:
            fname = videoWriteParams['folderName'] + '%0.05d.png'%frameNum
            fig1 = plt.gcf()
            plt.show()
            plt.draw()
            fig1.savefig(fname)

    plt.pause(0.05)


def resampleWeights(weights):
    N = weights.size
    u = (np.arange(0, N) + np.random.uniform(0, 1, N))/N
    cdf = np.cumsum(weights)
    ind = np.int_(np.zeros(np.shape(weights)))
    j = 0
    for k in xrange(0, N):
        while cdf[j] < u[k]:
            j += 1
        ind[k] = j
    return ind

# def mapCorrelation(im, x_im, y_im, vp, xs, ys):


def eul2rotm(eul):
    R = np.zeros((3,3))
    ct = np.cos(eul)
    st = np.sin(eul)
    R[0,0] = ct[1]*ct[0]
    R[0,1] = st[2]*st[1]*ct[0] - ct[2]*st[0]
    R[0,2] = ct[2]*st[1]*ct[0] + st[2]*st[0]
    R[1,0] = ct[1]*st[0]
    R[1,1] = st[2]*st[1]*st[0] + ct[2]*ct[0]
    R[1,2] = ct[2]*st[1]*st[0] - st[2]*ct[0]
    R[2,0] = -st[1]
    R[2,1] = st[2]*ct[1]
    R[2,2] = ct[2]*ct[1]
    return R


def texture_map(MAP, depth, rgb_image, pose, fc_rgb, fc_depth, h_angle, n_angle):
    tmap = np.zeros(Map['map'].shape)
    uv = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]))
    u = uv[0]
    v = uv[1]
    u = (u - np.max(u))/2.
    v = (v - np.max(v))/2.
    thresh = 0.1
    img = cv2.normalize(rgb_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    R = np.array([[0.999968551008760, 0.005899814450952, 0.005299922913182],
                  [-0.005894063933536, 0.999982024861347, -0.001099983885351],
                  [-0.005306317347155, 0.001068711207474, 0.999985350318977]])
    T = np.array([[52.268200000000000], [1.519200000000000], [-0.605900000000000]])

    # Convert to 3d points
    # depth = np.array([math.inf if i > 3000 else i for i in depth])
    # depth = np.array([math.inf if i < 500 else i for i in depth])
    # depth[depth > 3000] = np.inf
    # depth[depth < 500] = np.inf
    h = 1260
    x = (u*depth)/fc_depth[0]
    y = (v*depth)/fc_depth[1]
    xPoints = np.reshape(x, (1, x.size))
    yPoints = np.reshape(y, (1, y.size))
    zPoints = np.reshape(depth, (1, depth.size))

    # Find ground plane points
    h_correct = h + 70 * np.cos(h_angle)
    coeff = np.matmul(eul2rotm(np.array([0,0,h_angle])), np.array([[0],[1],[0]]))  # write eul2rotm
    val = np.sum(np.vstack((coeff,h_correct)) * np.vstack((xPoints,yPoints,zPoints,np.ones((1,xPoints.size)))),axis=0)
    GP = np.array([i for i in x if x[val]<thresh])
    GP = np.hstack((GP, y[val<thresh]))
    GP = np.hstack((GP, depth[val<thresh]))

    # Find corresponding RGB points
    GP_rgb = np.matmul(R, GP) + T
    u_rgb = np.ceil(fc_rgb[0] * (GP_rgb[0, :] / GP_rgb[2, :]) + img.shape[1] / 2)
    v_rgb = np.ceil(fc_rgb[1] * (GP_rgb[1, :] / GP_rgb[2, :]) + img.shape[0] / 2)

    indValid = (u_rgb < img.shape[1]) & (u_rgb > 0) & (v_rgb < img.shape[0]) & (v_rgb > 0)
    u_rgb = u_rgb[indValid]
    v_rgb = v_rgb[indValid]
    GP = GP[:,indValid]
    GP_rgb = GP_rgb[:,indValid]

    # Convert to Lidar frame
    GP_lidar = np.matmul(np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]), GP)

    # Convert xyz to map scale
    xindex = np.ceil(GP_lidar[0,:]/(1000*MAP['x_res']))
    yindex = np.ceil(GP_lidar[1,:]/(1000*MAP['y_res']))

    # Rotate based on pose
    ang = pose[2]
    index_new = np.matmul(np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]),
                          np.hstack((xindex, yindex)))
    xindex = index_new[0,:]
    yindex = index_new[1,:]

    # Translate based on pose
    xindex = np.ceil(xindex + pose[0]/MAP['x_res'] + MAP['grid_cols']/2)
    yindex = np.ceil(yindex + pose[1]/MAP['y_res'] + MAP['grid_rows']/2)








