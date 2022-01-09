# ----------
# Part Two
#
# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position.
#
# ----------
# GRADING
#
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import * # Check the matrix.py tab to see how this works.
import random

# This is the function you have to write. Note that measurement is a
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.


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

# the belowing circle radius, center calculation was referenced from http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
# different methods used to find the least squares circle fitting a set of 2D points (x,y).

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
                varlist[k].append(x)
        M = matrix([[sum(aa), sum(ab)], [sum(ab), sum(bb)]]).inverse() * matrix([[sum(aaa) + sum(abb)], [sum(bbb) + sum(aab)]])
        dce = [sum(self.xs) / len(self.xs) + M.value[0][0] / 2.0, sum(self.ys) / len(self.ys) + M.value[1][0] / 2.0]
        return dce

def estimate_next_pos(measurement, OTHER = None):

    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = []
        xs = []
        ys = []
        P = matrix([[1000.0, 0.0], [0.0, 1000.0]])
        OTHER = [measurements, xs, ys, P] # now I can keep track of history
    else: # not the first time, update my history
         measurements, xs, ys, P = OTHER # now I can always refer to these variables

    measurements.append(measurement)

    #apply kfilter to decrease noise
    # x = matrix([[measurement[0]], [measurement[1]]]) # initial state (location and velocity)
    # kfilter = KFilter()
    # x, P = kfilter.filter(measurements[-30:], x, P)
    # fx = x.value[0][0]
    # fy = x.value[1][0]
    # xy = (fx, fy)

    # the algorithm was referenced from http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    # different methods used to find the least squares circle fitting a set of 2D points (x,y).
    if len(xs) < 5:
        xs.append(measurement[0])
        ys.append(measurement[1])
        xy_estimate = measurement
    else:
        xs.append(measurement[0])
        ys.append(measurement[1])

        circle = Circle_1(xs, ys)
        dce = circle.implement()

        cir, beta, degree = [], [], []
        m_beta = 0
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
        for i in range(len(beta)):
            s_beta = s_beta + ((beta[i] - m_beta * i) % pi - (0 if beta[0] >= 0 else pi))
        s_beta = s_beta/len(beta)
        n_beta = (len(beta) + left) * m_beta + s_beta

        xy_estimate  = (dce[0] + Radius * cos(n_beta), dce[1] + Radius * sin(n_beta))

    OTHER[0] = measurements
    OTHER[1] = xs
    OTHER[2] = ys
    OTHER[3] = P

    return xy_estimate, OTHER


# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any
# information that you want.
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print "You got it right! It took you ", ctr, " steps to localize."
            localized = True
        if ctr == 1000:
            print "Sorry, it took you too many steps to localize the target."
    return localized

# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER = None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER: # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER


# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
# test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
# test_target = robot(x, y, ori, turn, dist)
# measurement_noise = 0.05 * test_target.distance
# test_target.set_noise(0.0, 0.0, measurement_noise)
# print demo_grading(estimate_next_pos, test_target)




