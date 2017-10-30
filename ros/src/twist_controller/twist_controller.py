from pid import PID
from lowpass import LowPassFilter
import rospy
import math

DEBUG = False

class Controller(object):
    def __init__(self, yaw_controller, accel_limit, decel_limit, weight):
        self.yaw_controller = yaw_controller
        self.acc_pid = PID(0.95, 0.0005, 0.075, 0, accel_limit)
        self.brake_pid = PID(500, 0.0005, 10.5, decel_limit * weight, 0)

        self.dbw_enabled = False
        self.timestamp = None

    def control(self, proposed_linear, proposed_angular, velocity, dbw_enabled):

        if proposed_linear < 0:
            proposed_linear = velocity

        if dbw_enabled is True and self.dbw_enabled is False:
            self.timestamp = None
            self.acc_pid.reset()
            self.brake_pid.reset()

        self.dbw_enabled = dbw_enabled

        if DEBUG:
            print "Proposed linear %.15f and proposed angular %.15f :::: velocity %.15f" % (proposed_linear, proposed_angular, velocity)

        err = proposed_linear - velocity
        timestamp = rospy.get_time()
        dt = timestamp - self.timestamp if self.timestamp is not None else 0.05

        throttle = self.acc_pid.step(err, dt)
        brake = math.fabs(self.brake_pid.step(err, dt))
        steer = self.yaw_controller.get_steering(proposed_linear, proposed_angular, velocity)

        self.timestamp = timestamp

        if DEBUG:
            print "%.15f :::: %.15f :::: %.15f" % (throttle, brake, steer)

        return throttle, brake, steer
