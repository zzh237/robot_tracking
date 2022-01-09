# ----------
# Part Three
#
# Now you'll actually track down and recover the runaway Traxbot.
# In this step, your speed will be about twice as fast the runaway bot,
# which means that your bot's distance parameter will be about twice that
# of the runaway. You can move less than this parameter if you'd
# like to slow down your bot near the end of the chase.
#
# ----------
# YOUR JOB
#
# Complete the next_move function. This function will give you access to
# the position and heading of your bot (the hunter); the most recent
# measurement received from the runaway bot (the target), the max distance
# your bot can move in a given timestep, and another variable, called
# OTHER, which you can use to keep track of information.
#
# Your function will return the amount you want your bot to turn, the
# distance you want your bot to move, and the OTHER variable, with any
# information you want to keep track of.
#
# ----------
# GRADING
#
# We will make repeated calls to your next_move function. After
# each call, we will move the hunter bot according to your instructions
# and compare its position to the target bot's true position
# As soon as the hunter is within 0.01 stepsizes of the target,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.
#
# As an added challenge, try to get to the target bot as quickly as
# possible.

from robot import *
from math import *
from matrix import *
import random


class KFilter:
    def __init__(self):

        self.a = matrix([[0.], [0.]]) # external motion
        self.F =  matrix([[1.0, 0.0], [0.0, 1.0]])
        self.H =  matrix([[1.0, 0.0], [0.0, 1.0]])
        self.R =  matrix([[0.1, 0.0], [0.0, 0.1]])
        self.I =  matrix([[1.0, 0.0], [0.0, 1.0]])

    def filter(self, measurements, x, P):
        for n in range(len(measurements)):
            # prediction
            x = (self.F * x) + self.a
            P = self.F * P * self.F.transpose()

            # measurement update
            Z = matrix([measurements[n]])
            y = Z.transpose() - (self.H * x)
            S = self.H * P * self.H.transpose() + self.R
            K = P * self.H.transpose() * S.inverse()
            x = x + (K * y)
            P = (self.I - (K * self.H)) * P
        return x, P

class Circle_1:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def implement(self):
        _x_ = sum(self.xs) / len(self.xs)
        _y_ = sum(self.ys) / len(self.ys)
        a, b = [], []
        for i in range(len(self.xs)):
            a.append(self.xs[i] - _x_)
            b.append(self.ys[i] - _y_)
        dic = {"a":a,"b":b}
        ab, aa, bb, aab, abb, aaa, bbb = [], [], [], [], [], [], []
        itemlist = ['ab', 'aa', 'bb', 'aab', 'abb', 'aaa', 'bbb']
        varlist = [ab, aa, bb, aab, abb, aaa, bbb]
        for k in range(len(itemlist)):
            t1 = list(itemlist[k])
            for i in range(len(self.xs)):
                x = 1
                for key in t1:
                    x *= dic[key][i]
                # exec "%s.append(%s)" % (itemlist[k], x)
            # print "aab", aab
                varlist[k].append(x)
        M = matrix([[sum(aa), sum(ab)], [sum(ab), sum(bb)]]).inverse() * matrix([[sum(aaa) + sum(abb)], [sum(bbb) + sum(aab)]])
        dce = [sum(self.xs) / len(self.xs) + M.value[0][0] / 2.0, sum(self.ys) / len(self.ys) + M.value[1][0] / 2.0]
        # A = array([[sum(aa), sum(ab)], [sum(ab), sum(bb)]])
        # B = array([sum(aaa) + sum(abb), sum(bbb) + sum(aab)])/2
        # uc, vc = linalg.solve(A, B)
        # xc_1 = _x_ + uc
        # yc_1 = _y_ + vc
        # dce = [xc_1, yc_1]
        # Ri = calc_R(*dce)
        # Radius = Ri.mean()
        # residu    = sum((Ri - Radius)**2)
        return dce

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = []
        hunter_positions = []
        hunter_headings = []
        xs = []
        ys = []
        P = matrix([[1000.0, 0.0], [0.0, 1000.0]])
        OTHER = [measurements, hunter_positions, hunter_headings, xs, ys, P] # now I can keep track of history
    else: # not the first time, update my history
        measurements, hunter_positions, hunter_headings, xs, ys, P = OTHER # now I can always refer to these variables


    measurements.append(target_measurement)
    hunter_positions.append(hunter_position)
    hunter_headings.append(hunter_heading)

    #apply kfilter to decrease noise
    # x = matrix([[target_measurement[0]], [target_measurement[1]]]) # initial state (location and velocity)
    # kfilter = KFilter()
    # x, P = kfilter.filter(measurements[-30:], x, P)
    # fx = x.value[0][0]
    # fy = x.value[1][0]
    # xy = (fx, fy)

    # the algorithm was referenced from http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    # different methods used to find the least squares circle fitting a set of 2D points (x,y).

    if len(xs) < 5:
        xs.append(target_measurement[0])
        ys.append(target_measurement[1])
        pred_position  =target_measurement
    else:
        xs.append(target_measurement[0])
        ys.append(target_measurement[1])

        _x_ = sum(xs) / len(xs)
        _y_ = sum(ys) / len(ys)

        circle = Circle_1(xs, ys)
        dce = circle.implement()

        cir, beta, degree = [], [], []
        m_beta, m_degree = 0, 0
        for i in range(len(xs)):
            beta.append(atan2(ys[i] - dce[1], xs[i] - dce[0]))
            degree.append(degrees(beta[i]))
            m_beta = m_beta + (angle_trunc(beta[i] - beta[i - 1]) if i > 0 else 0)
            cir.append(sqrt((xs[i]-dce[0])**2 + (ys[i]-dce[1])**2))
        m_beta /= (len(beta) - 1)
        Radius = sum(cir) / len(cir)
        m_degree = degrees(m_beta)

        s_beta = 0
        count = 4
        left = count - len(OTHER[0]) % count - 1
        for i in range(len(xs)):
            s_beta = s_beta + ((beta[i] - m_beta * i) % pi - (0 if beta[0] >= 0 else pi))
        s_beta = s_beta/len(beta)
        n_beta = (len(beta) + left) * m_beta + s_beta
        pred_position  = (dce[0] + Radius * cos(n_beta), dce[1] + Radius * sin(n_beta))

    OTHER[0] = measurements
    OTHER[1] = hunter_positions
    OTHER[2] = hunter_headings
    OTHER[3] = xs
    OTHER[4] = ys
    OTHER[5] = P

    heading_to_target = get_heading(hunter_position, pred_position )
    turning = angle_trunc(heading_to_target - hunter_heading)
    distance = sqrt((pred_position [0] - hunter_position[0])**2 + (pred_position [1] - hunter_position[1])**2)

    return turning, distance, OTHER

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 1.94 * target_bot.distance # 1.94 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught, ctr

def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all
    the target measurements, hunter positions, and hunter headings over time, but it doesn't
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables

    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER

# target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
# measurement_noise = .05*target.distance
# target.set_noise(0.0, 0.0, measurement_noise)
#
# hunter = robot(-10.0, -10.0, 0.0)

# print demo_grading(hunter, target, next_move)
# import numpy as np
# import timeit
# def curr_time_millis():
#     return 1000 * timeit.default_timer()
# # This is how we create a target bot. Check the robot.py file to understand
# # How the robot class behaves.
# failed_ctr = 0
# steps_list = []
# time_step = []
# N = 100
# for trial in range(N):
#     xt = random.uniform(-20,20)
#     yt = random.uniform(-20,20)
#     xh = random.uniform(-20,20)
#     yh = random.uniform(-20,20)
#     turn = random.uniform((10*pi)/180,(50*pi)/180)
#     test = random.uniform(-1,1)
#     if test < 0:
#         turn *= -1
#     orih = random.uniform(-pi,pi)
#     orit = random.uniform(-pi,pi)
#     dist = random.uniform(1.,5.)
#     #def __init__(self, x = 0.0, y = 0.0, heading = 0.0, turning = 2*pi/10, distance = 1.0):
#     #target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
#     target = robot(xt,yt,orit,turn,dist)
#     measurement_noise = .05*target.distance
#     target.set_noise(0.0, 0.0, measurement_noise)
#     #hunter = robot(-10.0, -10.0, 0.0)
#     hunter = robot(xh,yh,orih)
#     current_time = curr_time_millis()
#     res, ctr = demo_grading(hunter, target, next_move)
#     time_step.append(curr_time_millis() - current_time)
#     if res == False:
#         failed_ctr += 1
#     else:
#         steps_list.append(ctr)
#
# print "Failed attempts ==> ", failed_ctr, "out of {} trials".format(N)
# print "Average time taken (ms)", np.mean(time_step)
# print "Minimum time taken (ms)", np.min(time_step)
# print "Maximum time taken (ms)", np.max(time_step)
# if len(steps_list) > 0:
#    print "Average steps taken ", np.mean(steps_list)
#    print "Mininum steps taken ", np.min(steps_list)
#    print "Maximum steps taken ", np.max(steps_list)


