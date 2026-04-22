#!/usr/bin/env python

import numpy as np
import sys
import cv2
import time
import rospy
import tf2_ros
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from tf.transformations import euler_from_quaternion


class OccupancyGridMap:
    def __init__(self):
        #Topics & Subs, Pubs
        # Read paramters form params.yaml
        lidarscan_topic = rospy.get_param('~scan_topic')
        odom_topic = rospy.get_param('~odom_topic')

        self.t_prev = rospy.get_time()
        self.max_lidar_range = rospy.get_param('~scan_range')
        self.scan_beams = rospy.get_param('~scan_beams')

        # Read the map parameters from *.yaml file
        # Grid dimensions and resolution
        self.grid_cols   = rospy.get_param('~map_width')       
        self.grid_rows   = rospy.get_param('~map_height')      
        self.cell_size   = rospy.get_param('~map_res')         
        self.prob_hit    = rospy.get_param('~p_occ')          
        self.prob_miss   = rospy.get_param('~p_free')          
        self.obj_size    = rospy.get_param('~object_size')
        self.fixed_frame = rospy.get_param('~odom_frame')
        grid_topic       = rospy.get_param('~occ_map_topic')

        # limit log-odds to a safe range to avoid float overflow
        self.LOGODDS_LIMIT = 50.0

        # Place the grid origin so the robot starts near the centre
        self.origin_x = -(self.grid_cols * self.cell_size) / 2.0
        self.origin_y = -(self.grid_rows * self.cell_size) / 2.0

        # Rigid offset of the LiDAR from the robot base frame
        self.lidar_dx = 0.01286
        self.lidar_dy = 0.0

        # log-odds equivalents of the sensor probabilities
        self.lo_hit   = math.log(self.prob_hit  / (1.0 - self.prob_hit))
        self.lo_miss  = math.log(self.prob_miss / (1.0 - self.prob_miss))
        self.lo_prior = 0.0   # log(0.5/0.5) = 0

        # Allocate the log-odds grid; zero = unknown
        self.logodds_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        # Robot pose placeholders updated by odometry
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self.got_odom  = False

        # Build and populate the OccupancyGrid ROS message header/metadata
        self.map_occ_grid_msg = OccupancyGrid()
        self.map_occ_grid_msg.header.frame_id            = self.fixed_frame
        self.map_occ_grid_msg.header.stamp               = rospy.Time.now()
        self.map_occ_grid_msg.info.resolution            = self.cell_size
        self.map_occ_grid_msg.info.width                 = self.grid_cols
        self.map_occ_grid_msg.info.height                = self.grid_rows
        self.map_occ_grid_msg.info.origin.position.x    = self.origin_x
        self.map_occ_grid_msg.info.origin.position.y    = self.origin_y
        self.map_occ_grid_msg.info.origin.position.z    = 0.0
        self.map_occ_grid_msg.info.origin.orientation.x = 0.0
        self.map_occ_grid_msg.info.origin.orientation.y = 0.0
        self.map_occ_grid_msg.info.origin.orientation.z = 0.0
        self.map_occ_grid_msg.info.origin.orientation.w = 1.0

        # All cells start as unknown (-1 in the ROS convention)
        self.map_occ_grid_msg.data = [-1] * (self.grid_cols * self.grid_rows)

        # Initialize the cell occupancy probabilites to 0.5 (unknown) with all cell data in Occupancy Grid Message set to unknown
        # Subscribe to Lidar scan and odometry topics with corresponding lidar_callback() and odometry_callback() functions
        self.sub_odom  = rospy.Subscriber(odom_topic,      Odometry,  self.odom_callback,  queue_size=1)
        self.sub_lidar = rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)

        # Create a publisher for the Occupancy Grid Map
        self.map_pub   = rospy.Publisher(grid_topic,        OccupancyGrid, queue_size=1)

        rospy.loginfo("OccupancyGridMap ready.")

    def odom_callback(self, odom_msg):
        """Cache the latest robot pose from wheel odometry."""
        self.robot_x = odom_msg.pose.pose.position.x
        self.robot_y = odom_msg.pose.pose.position.y

        q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw
        self.got_odom  = True

    # lidar_callback () uses the current LiDAR scan and Wheel Odometry data to uddate and publish the Grid Occupancy map 
    def lidar_callback(self, data):
        """
        Update the occupancy grid from one LiDAR scan using the
        log-odds Bayesian update rule.

        For every grid cell:
          1. Find the LiDAR beam whose direction best matches the
             vector from the sensor to that cell.
          2. Apply the inverse sensor model:
               - Occupied : beam endpoint lands inside the cell  -> lo_hit
               - Free     : beam passes beyond the cell          -> lo_miss
               - Unknown  : beam stops before the cell           -> no update
          3. Accumulate into the log-odds grid and convert to
             a [0, 100] probability for publishing.
        """

        if not self.got_odom:
            return

        # AH: Current robot pose
        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw

        # CW: LiDAR position in the world frame (offset from base_link)
        sx = rx + self.lidar_dx * math.cos(ryaw)
        sy = ry + self.lidar_dx * math.sin(ryaw)
        sensor_heading = ryaw + math.pi   # TS: sensor forward direction in world frame

        # AH, CW: Scan metadata
        a_min  = data.angle_min
        a_inc  = data.angle_increment
        n_rays = len(data.ranges)
        scans  = data.ranges

        # TS - Half cell diagonal used to decide if an endpoint is "inside" a cell
        half_c = self.cell_size / 2.0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cx = self.origin_x + (c + 0.5) * self.cell_size
                cy = self.origin_y + (r + 0.5) * self.cell_size

                vx = cx - sx
                vy = cy - sy
                dist = math.sqrt(vx * vx + vy * vy)

                if dist < 1e-6:
                    continue

                # Angle to cell in sensor frame
                angle_world  = math.atan2(vy, vx)
                angle_sensor = angle_world - sensor_heading
                angle_sensor = math.atan2(math.sin(angle_sensor),
                                          math.cos(angle_sensor))

                # Skip if outside the scanner's field of view
                if angle_sensor < a_min or angle_sensor > data.angle_max:
                    continue

                # Nearest beam index
                idx = int(round((angle_sensor - a_min) / a_inc))
                idx = max(0, min(n_rays - 1, idx))

                meas = scans[idx]

                if math.isnan(meas) or math.isinf(meas) or meas <= 0.0:
                    continue

                beam_angle_world = (a_min + idx * a_inc) + sensor_heading
                ex = sx + meas * math.cos(beam_angle_world)
                ey = sy + meas * math.sin(beam_angle_world)

                # Inverse sensor model
                if meas >= self.max_lidar_range:
                    # Max-range return: only label cells closer than max as free
                    if dist < self.max_lidar_range:
                        lo_update = self.lo_miss
                    else:
                        continue
                else:
                    # Check if the beam endpoint falls inside this cell (bounding box test)
                    hit_in_cell = (abs(ex - cx) <= half_c and abs(ey - cy) <= half_c)

                    if hit_in_cell:
                        lo_update = self.lo_hit           # obstacle detected here
                    elif meas > dist:
                        lo_update = self.lo_miss          # beam passed through - free
                    else:
                        continue                          # beam stopped before cell - no info

                # Additive log-odds update (prior = 0 cancels out)
                self.logodds_grid[r][c] += lo_update
                self.logodds_grid[r][c] = max(
                    -self.LOGODDS_LIMIT,
                    min(self.LOGODDS_LIMIT, self.logodds_grid[r][c])
                )

        # AH - Convert log-odds to ROS occupancy values
        out = [-1] * (self.grid_cols * self.grid_rows)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                lo = self.logodds_grid[r][c]

                if lo == 0.0:
                    out[r * self.grid_cols + c] = -1   # still at prior means unknown
                else:
                    lo_cl = max(-self.LOGODDS_LIMIT, min(self.LOGODDS_LIMIT, lo))
                    prob  = 1.0 - 1.0 / (1.0 + math.exp(lo_cl))

                    if prob > self.prob_hit:
                        out[r * self.grid_cols + c] = 100   # occupied
                    elif prob < self.prob_miss:
                        out[r * self.grid_cols + c] = 0     # free
                    else:
                        out[r * self.grid_cols + c] = -1    # uncertain

        self.map_occ_grid_msg.data = out
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_pub.publish(self.map_occ_grid_msg)


def main(args):
    rospy.init_node("occupancygridmap", anonymous=True)
    OccupancyGridMap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
