from pid import PID
from lowpass import LowPassFilter
import rospy
import math

class Controller(object):
    def __init__(self, yaw_controller, accel_limit, decel_limit):
        self.yaw_controller = yaw_controller
        self.pid = PID(2, 0.05, 0.1, decel_limit, accel_limit)
        self.lp_filter = LowPassFilter(10., 1.)

        self.dbw_enabled = False
        self.timestamp = None

    def control(self, proposed_linear, proposed_angular, velocity, dbw_enabled):
        if dbw_enabled is True and self.dbw_enabled is False:
            self.timestamp = None
            self.pid.reset()

        self.dbw_enabled = dbw_enabled
        proposed_linear = self.lp_filter.filt(proposed_linear)

        err = proposed_linear - velocity
        timestamp = rospy.get_time()
        dt = timestamp - self.timestamp if self.timestamp is not None else 0.05

        a = self.pid.step(err, dt)
        throttle, brake = (a, 0.) if a > 0. else (0., math.fabs(a))
        steer = self.yaw_controller.get_steering(proposed_linear, proposed_angular, velocity)

        self.timestamp = timestamp

        return throttle, brake, steer
