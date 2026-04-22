#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
 
class WallFollow:
    def __init__(self):
        self.lidarscan_topic = rospy.get_param("~lidarscan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.drive_topic = rospy.get_param("~nav_drive_topic", "/nav")
 
 
        # Beam angles for wall detection (degrees)
        # SIMULATOR (forward-facing laser): bl=90, al=40, br=270, ar=320
        # REAL CAR  (backward-facing laser): bl=270, al=220, br=90, ar=140
        self.bl_deg = rospy.get_param("~bl_angle_deg")
        self.al_deg = rospy.get_param("~al_angle_deg")
        self.br_deg = rospy.get_param("~br_angle_deg")
        self.ar_deg = rospy.get_param("~ar_angle_deg")
 
        self.kp = rospy.get_param("~kp_linefollow")
        self.kd = rospy.get_param("~kd_linefollow")
        self.vs_des = rospy.get_param("~vs_des")
        self.delta_theta_deg = rospy.get_param("~front_view_angle_deg")
        self.d_stop = rospy.get_param("~d_stop")
        self.d_tau = rospy.get_param("~d_tau")
        self.delta_max = rospy.get_param("~delta_max")
        self.dlr_des = rospy.get_param("~dlr_des")
        self.wheelbase = rospy.get_param("~wheelbase")
        self.min_speed_for_control = rospy.get_param("~min_speed_for_control")
        self.front_angle_deg = rospy.get_param("~front_angle_deg")
        self.front_angle = math.radians(self.front_angle_deg)
 
        self.bl = math.radians(self.bl_deg)
        self.al = math.radians(self.al_deg)
        self.br = math.radians(self.br_deg)
        self.ar = math.radians(self.ar_deg)
        self.delta_theta = math.radians(self.delta_theta_deg)
        self.vel = 0.0
 
        rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        rospy.loginfo("navigation_line_following started")
        rospy.loginfo("  Publishing to: %s", self.drive_topic)
        rospy.loginfo("  bl=%.0f al=%.0f br=%.0f ar=%.0f",
                      self.bl_deg, self.al_deg, self.br_deg, self.ar_deg)
        rospy.loginfo("  kp=%.2f kd=%.2f v_des=%.2f front=%.1f",
                      self.kp, self.kd, self.vs_des, self.front_angle_deg)
 
    @staticmethod
    def wrap_to_pi(angle):
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
 
    @staticmethod
    def clip(x, lo, hi):
        return max(lo, min(hi, x))
 
    def get_range_at_angle(self, scan_msg, angle_rad):
        angle = self.wrap_to_pi(angle_rad)
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        inc = scan_msg.angle_increment
        n = len(scan_msg.ranges)
        if angle < angle_min:
            angle += 2.0 * math.pi
        if angle > angle_max:
            angle -= 2.0 * math.pi
        idx = int(round((angle - angle_min) / inc))
        idx = max(0, min(n - 1, idx))
        r = scan_msg.ranges[idx]
        if np.isnan(r) or np.isinf(r):
            return scan_msg.range_max
        return self.clip(r, scan_msg.range_min, scan_msg.range_max)
 
    def get_front_obstacle_distance(self, scan_msg):
        center = self.front_angle
        half = self.delta_theta / 2.0
        min_dist = scan_msg.range_max
        for i, r in enumerate(scan_msg.ranges):
            if np.isnan(r) or np.isinf(r):
                continue
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            diff = abs(self.wrap_to_pi(beam_angle - center))
            if diff <= half:
                rr = self.clip(r, scan_msg.range_min, scan_msg.range_max)
                if rr < min_dist:
                    min_dist = rr
        return min_dist
 
    def wall_from_beams(self, scan_msg, angle_b, angle_a):
 
        r_b = self.get_range_at_angle(scan_msg, angle_b)
        r_a = self.get_range_at_angle(scan_msg, angle_a)
 
     
        ab = self.wrap_to_pi(angle_b)
        aa = self.wrap_to_pi(angle_a)
        xb = r_b * math.cos(ab)
        yb = r_b * math.sin(ab)
        xa = r_a * math.cos(aa)
        ya = r_a * math.sin(aa)
 
      
        dx = xa - xb
        dy = ya - yb
        wall_len = math.sqrt(dx * dx + dy * dy)
        if wall_len < 1e-6:
            return None
 
        # Perpendicular distance from origin to wall line
        cross = dx * yb - dy * xb
        dist = abs(cross) / wall_len
 
        # Normalized wall direction
        tx = dx / wall_len
        ty = dy / wall_len
 
        # Ensure wall direction points "forward" (tx > 0)
        if tx < 0:
            tx = -tx
            ty = -ty
 
        return dist, tx, ty
 
    def publish_drive(self, speed_cmd, steer_cmd):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = speed_cmd
        msg.drive.steering_angle = steer_cmd
        self.drive_pub.publish(msg)
 
    def lidar_callback(self, scan_msg):
        # Compute wall geometry using cartesian approach
        right_wall = self.wall_from_beams(scan_msg, self.br, self.ar)
        left_wall = self.wall_from_beams(scan_msg, self.bl, self.al)
 
        if right_wall is None or left_wall is None:
            self.publish_drive(0.0, 0.0)
            return
 
        dr, txr, tyr = right_wall
        dl, txl, tyl = left_wall
 
        dr = self.clip(dr, 0.05, 10.0)
        dl = self.clip(dl, 0.05, 10.0)
 
        # d_lr = d_l - d_r (positive means car is closer to right wall)
        dlr = dl - dr
        dlr_tilde = dlr - self.dlr_des
 
        vs = self.vel
 
        # Rate of change of d_lr from wall geometry:
        # d_dot_r = -vs * tyr (from inward normal of right wall)
        # d_dot_l = vs * tyl  (from inward normal of left wall)
        # d_dot_lr = d_dot_l - d_dot_r = vs * (tyl + tyr)
        dlr_dot = vs * (tyl + tyr)
 
        # cos(alpha) from wall surface direction
        cos_alpha_r = txr
        cos_alpha_l = txl
        cos_sum = cos_alpha_r + cos_alpha_l
 
        # Feedback-linearizing + PD controller (Eq 8)
        vs_eff = max(abs(vs), self.min_speed_for_control)
        denom = (vs_eff ** 2) * cos_sum
 
        if abs(denom) < 1e-6:
            delta = 0.0
        else:
            u = -self.kp * dlr_tilde - self.kd * dlr_dot
            delta = math.atan((-self.wheelbase / denom) * u)
 
        delta_c = self.clip(delta, -self.delta_max, self.delta_max)
 
        # Velocity control (Eq 20)
        d_ob = self.get_front_obstacle_distance(scan_msg)
        expo_arg = -max(d_ob - self.d_stop, 0.0) / self.d_tau
        vs_c = self.vs_des * (1.0 - math.exp(expo_arg))
 
        self.publish_drive(vs_c, delta_c)
 
    def odom_callback(self, odom_msg):
        self.vel = odom_msg.twist.twist.linear.x
 
def main(args):
    rospy.init_node("navigation_line_following", anonymous=True)
    WallFollow()
    rospy.sleep(0.1)
    rospy.spin()
 
if __name__ == '__main__':
    main(sys.argv)