#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
 
class GapBarrier(object):
    def __init__(self):
        self.lidarscan_topic = rospy.get_param("~lidarscan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        # Keep the default on the existing nav channel so the current mux setup works.
        #self.drive_topic = rospy.get_param(
           # "~drive_topic",
           # rospy.get_param("~nav_drive_topic", "/nav"),
        #)

        self.drive_topic = rospy.get_param("~nav_drive_topic", "/nav")
        self.wheelbase = rospy.get_param("~wheelbase")
        self.width = rospy.get_param("~width")
        self.delta_max = rospy.get_param("~delta_max")
        self.vs_des = rospy.get_param("~vs_des")
        self.min_speed_for_control = rospy.get_param("~min_speed_for_control")
        self.front_angle_deg = rospy.get_param("~front_angle_deg")
        self.fov_deg = rospy.get_param("~fov_deg")
        self.front_view_angle_deg = rospy.get_param("~front_view_angle_deg")
        self.barrier_sector_l_deg = rospy.get_param("~barrier_sector_l_deg")
        self.barrier_sector_r_deg = rospy.get_param("~barrier_sector_r_deg")
        self.safe_dist = rospy.get_param("~safe_dist")
        self.d_stop = rospy.get_param("~d_stop")
        self.d_tau = rospy.get_param("~d_tau")
        self.k_heading = rospy.get_param("~k_heading")
        self.k_center = rospy.get_param("~k_center")
        self.clearance_margin = rospy.get_param("~clearance_margin")
        self.default_half_width = rospy.get_param("~default_half_width")
        self.barrier_lookahead = rospy.get_param("~barrier_lookahead")
        self.smoothing_window = int(rospy.get_param("~smoothing_window"))
        self.min_gap_beams = int(rospy.get_param("~min_gap_beams"))
        self.angle_bias = rospy.get_param("~angle_bias")
        self.steer_filter = rospy.get_param("~steer_filter")
        self.creep_speed = rospy.get_param("~creep_speed")
        self.front_angle = math.radians(self.front_angle_deg)
        self.fov = math.radians(self.fov_deg)
        self.front_view_angle = math.radians(self.front_view_angle_deg)
        self.barrier_sector_l = math.radians(self.barrier_sector_l_deg)
        self.barrier_sector_r = math.radians(self.barrier_sector_r_deg)
        self.vehicle_half_width = 0.5 * self.width
        self.vel = 0.0
        self.prev_steer = 0.0
        self.prev_left_offset = self.default_half_width
        self.prev_right_offset = -self.default_half_width
        rospy.Subscriber(self.lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        rospy.loginfo("navigation_virtual_barrier started")
        rospy.loginfo("  drive topic: %s", self.drive_topic)
        rospy.loginfo(
            "  fov=%.1f safe_dist=%.2f d_stop=%.2f k_heading=%.2f k_center=%.2f",
            self.fov_deg,
            self.safe_dist,
            self.d_stop,
            self.k_heading,
            self.k_center,
        )

    @staticmethod
    def wrap_to_pi(angle):
        while angle >= math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def clip(value, lo, hi):
        return max(lo, min(hi, value))

    def publish_drive(self, speed_cmd, steer_cmd):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = speed_cmd
        msg.drive.steering_angle = steer_cmd
        self.drive_pub.publish(msg)

    def preprocess_lidar(self, scan_msg):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        if ranges.size == 0:
            return None
        invalid = np.logical_not(np.isfinite(ranges))
        ranges[invalid] = scan_msg.range_max
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)
        beam_ids = np.arange(ranges.size, dtype=np.int32)
        angles = scan_msg.angle_min + beam_ids * scan_msg.angle_increment
        rel_angles = np.arctan2(
            np.sin(angles - self.front_angle),
            np.cos(angles - self.front_angle),
        )
        sector_mask = np.abs(rel_angles) <= 0.5 * self.fov
        sector_ids = np.where(sector_mask)[0]
        if sector_ids.size == 0:
            return None
        order = np.argsort(rel_angles[sector_ids])
        sector_ids = sector_ids[order]
        sector_angles = angles[sector_ids]
        sector_rel_angles = rel_angles[sector_ids]
        sector_ranges = ranges[sector_ids]
        if self.smoothing_window > 1 and sector_ranges.size >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window, dtype=np.float64)
            kernel /= np.sum(kernel)
            smooth_ranges = np.convolve(sector_ranges, kernel, mode="same")
        else:
            smooth_ranges = sector_ranges.copy()
        proc_ranges = smooth_ranges.copy()
        beam_width = abs(scan_msg.angle_increment)
        inflated_half_width = self.vehicle_half_width + self.clearance_margin
        # Shrink gaps by the vehicle footprint so the chosen corridor is actually drivable.
        for i, dist in enumerate(smooth_ranges):
            if dist <= scan_msg.range_min:
                proc_ranges[i] = 0.0
                continue
            spread = int(math.ceil(math.atan2(inflated_half_width, max(dist, 0.05)) / beam_width))
            start_i = max(0, i - spread)
            end_i = min(proc_ranges.size, i + spread + 1)
            proc_ranges[start_i:end_i] = np.minimum(proc_ranges[start_i:end_i], dist)
        proc_ranges[proc_ranges < self.safe_dist] = 0.0
        return {
            "sector_ids": sector_ids,
            "angles": sector_angles,
            "rel_angles": sector_rel_angles,
            "raw_ranges": sector_ranges,
            "proc_ranges": proc_ranges,
            "beam_width": beam_width,
        }

    def find_max_gap(self, proc_ranges):
        best_start = -1
        best_end = -1
        best_len = 0
        curr_start = -1
        for i in range(proc_ranges.size):
            if proc_ranges[i] > 0.0:
                if curr_start < 0:
                    curr_start = i
            elif curr_start >= 0:
                curr_end = i - 1
                curr_len = curr_end - curr_start + 1
                if curr_len > best_len:
                    best_len = curr_len
                    best_start = curr_start
                    best_end = curr_end
                curr_start = -1
        if curr_start >= 0:
            curr_end = proc_ranges.size - 1
            curr_len = curr_end - curr_start + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
                best_end = curr_end
        if best_len < self.min_gap_beams:
            return None
        return best_start, best_end

    def find_best_point(self, start_i, end_i, proc_ranges, rel_angles):
        gap_ranges = proc_ranges[start_i : end_i + 1]
        gap_angles = rel_angles[start_i : end_i + 1]
        if gap_ranges.size == 0:
            return None
        angle_scale = np.maximum(0.0, 1.0 - self.angle_bias * np.abs(gap_angles) / max(0.5 * self.fov, 1e-6))
        scores = gap_ranges * angle_scale
        top_k = min(5, scores.size)
        top_indices = np.argsort(scores)[-top_k:]
        best_local = int(round(np.mean(top_indices)))
        return start_i + best_local

    def getWalls(self, left_obstacles, right_obstacles, wl0=None, wr0=None, alpha=0.0):
        heading = np.array([math.cos(alpha), math.sin(alpha)])
        normal = np.array([-math.sin(alpha), math.cos(alpha)])
        left_offset = None
        right_offset = None
        if left_obstacles.shape[0] > 0:
            left_s = np.dot(left_obstacles, normal)
            left_t = np.dot(left_obstacles, heading)
            left_mask = np.logical_and(left_s > 0.0, np.logical_and(left_t > 0.05, left_t < self.barrier_lookahead))
            if np.any(left_mask):
                left_offset = float(np.percentile(left_s[left_mask], 20.0))
        if right_obstacles.shape[0] > 0:
            right_s = np.dot(right_obstacles, normal)
            right_t = np.dot(right_obstacles, heading)
            right_mask = np.logical_and(right_s < 0.0, np.logical_and(right_t > 0.05, right_t < self.barrier_lookahead))
            if np.any(right_mask):
                right_offset = -float(np.percentile(np.abs(right_s[right_mask]), 20.0))
        if left_offset is None and right_offset is None:
            return None
        if left_offset is None:
            left_offset = max(self.vehicle_half_width + self.clearance_margin, -right_offset)
        if right_offset is None:
            right_offset = -max(self.vehicle_half_width + self.clearance_margin, left_offset)
        left_offset = 0.7 * self.prev_left_offset + 0.3 * left_offset
        right_offset = 0.7 * self.prev_right_offset + 0.3 * right_offset
        corridor_half_width = 0.5 * (left_offset - right_offset)
        center_offset = 0.5 * (left_offset + right_offset)
        self.prev_left_offset = left_offset
        self.prev_right_offset = right_offset
        return {
            "left_offset": left_offset,
            "right_offset": right_offset,
            "center_offset": center_offset,
            "half_width": corridor_half_width,
            "heading": alpha,
        }

    def get_front_obstacle_distance(self, scan_msg):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        if ranges.size == 0:
            return scan_msg.range_max
        invalid = np.logical_not(np.isfinite(ranges))
        ranges[invalid] = scan_msg.range_max
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)
        beam_ids = np.arange(ranges.size, dtype=np.int32)
        angles = scan_msg.angle_min + beam_ids * scan_msg.angle_increment
        rel_angles = np.arctan2(
            np.sin(angles - self.front_angle),
            np.cos(angles - self.front_angle),
        )
        mask = np.abs(rel_angles) <= 0.5 * self.front_view_angle
        if not np.any(mask):
            return scan_msg.range_max
        return float(np.min(ranges[mask]))

    def fallback_drive(self, front_dist, sector_rel_angles, raw_ranges):
        left_mask = sector_rel_angles > 0.0
        right_mask = sector_rel_angles < 0.0
        left_clear = np.mean(raw_ranges[left_mask]) if np.any(left_mask) else front_dist
        right_clear = np.mean(raw_ranges[right_mask]) if np.any(right_mask) else front_dist
        steer_cmd = 0.6 * (left_clear - right_clear)
        steer_cmd = self.clip(steer_cmd, -self.delta_max, self.delta_max)
        if front_dist < self.d_stop:
            speed_cmd = 0.0
        else:
            speed_cmd = min(self.creep_speed, self.vs_des)
        self.prev_steer = steer_cmd
        self.publish_drive(speed_cmd, steer_cmd)

    def lidar_callback(self, scan_msg):
        lidar_data = self.preprocess_lidar(scan_msg)
        if lidar_data is None:
            self.publish_drive(0.0, 0.0)
            return
        proc_ranges = lidar_data["proc_ranges"]
        raw_ranges = lidar_data["raw_ranges"]
        rel_angles = lidar_data["rel_angles"]
        abs_angles = lidar_data["angles"]
        gap = self.find_max_gap(proc_ranges)
        front_dist = self.get_front_obstacle_distance(scan_msg)
        if gap is None:
            self.fallback_drive(front_dist, rel_angles, raw_ranges)
            return
        best_idx = self.find_best_point(gap[0], gap[1], proc_ranges, rel_angles)
        if best_idx is None:
            self.fallback_drive(front_dist, rel_angles, raw_ranges)
            return
        desired_heading = abs_angles[best_idx]
        x_pts = raw_ranges * np.cos(abs_angles)
        y_pts = raw_ranges * np.sin(abs_angles)
        points = np.column_stack((x_pts, y_pts))
        rel_to_heading = np.arctan2(
            np.sin(abs_angles - desired_heading),
            np.cos(abs_angles - desired_heading),
        )
        point_mask = np.logical_and(raw_ranges > 0.05, raw_ranges < self.barrier_lookahead + self.default_half_width)
        left_mask = np.logical_and(point_mask, np.logical_and(rel_to_heading >= 0.0, rel_to_heading <= self.barrier_sector_l))
        right_mask = np.logical_and(point_mask, np.logical_and(rel_to_heading <= 0.0, rel_to_heading >= -self.barrier_sector_r))
        walls = self.getWalls(points[left_mask], points[right_mask], None, None, desired_heading)
        if walls is None:
            self.fallback_drive(front_dist, rel_angles, raw_ranges)
            return
        heading_error = self.wrap_to_pi(desired_heading - self.front_angle)
        center_error = walls["center_offset"]
        corridor_half_width = max(walls["half_width"], self.vehicle_half_width + 0.01)
        raw_steer = self.k_heading * heading_error + self.k_center * center_error
        steer_cmd = self.steer_filter * self.prev_steer + (1.0 - self.steer_filter) * raw_steer
        steer_cmd = self.clip(steer_cmd, -self.delta_max, self.delta_max)
        self.prev_steer = steer_cmd
        clearance_scale = self.clip(
            (corridor_half_width - self.vehicle_half_width) / max(self.default_half_width - self.vehicle_half_width, 1e-3),
            0.15,
            1.0,
        )
        steering_scale = self.clip(1.0 - 0.75 * abs(steer_cmd) / max(self.delta_max, 1e-3), 0.25, 1.0)
        exp_arg = -max(front_dist - self.d_stop, 0.0) / max(self.d_tau, 1e-3)
        speed_nominal = self.vs_des * (1.0 - math.exp(exp_arg))
        speed_cmd = speed_nominal * clearance_scale * steering_scale
        if front_dist < self.d_stop:
            speed_cmd = self.creep_speed if abs(heading_error) > math.radians(8.0) else 0.0
        speed_cmd = self.clip(speed_cmd, 0.0, self.vs_des)
        if speed_cmd > 0.0:
            speed_cmd = max(speed_cmd, self.min_speed_for_control)

        self.publish_drive(speed_cmd, steer_cmd)

    def odom_callback(self, odom_msg):
        self.vel = odom_msg.twist.twist.linear.x

def main(args):
    rospy.init_node("GapBarrier_node", anonymous=True)
    GapBarrier()
    rospy.sleep(0.1)
    rospy.spin()
 
if __name__ == "__main__":
    main(sys.argv)