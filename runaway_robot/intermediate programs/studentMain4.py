# ----------
# Part Four
#
# Again, you'll track down and recover the runaway Traxbot.
# But this time, your speed will be about the same as the runaway bot.
# This may require more careful planning than you used last time.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time.
#
# ----------
# GRADING
#
# Same as part 3. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrix import *
import random

def kfilter(measurements, x, P):
    for n in range(len(measurements)):

        # prediction
        x = (F * x) + a
        P = F * P * F.transpose()

        # measurement update
        Z = matrix([measurements[n]])
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P
    return x, P

initial_xy = [4., 12.]

dt = 0.1

x = matrix([[0], [0]]) # initial state (location and velocity)
a = matrix([[0.], [0.]]) # external motion

F =  matrix([[1.0, 0.0], [0.0, 1.0]])
H =  matrix([[1.0, 0.0], [0.0, 1.0]])
R =  matrix([[0.1, 0.0], [0.0, 0.1]])
I =  matrix([[1.0, 0.0], [0.0, 1.0]])

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # print "program s_"
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = []
        hunter_headings = []
        xs = []
        ys = []
        P = matrix([[1000.0, 0.0], [0.0, 1000.0]])
        OTHER = [measurements, hunter_positions, hunter_headings, xs, ys, P] # now I can keep track of history
    else: # not the first time, update my history
        measurements, hunter_positions, hunter_headings, xs, ys, P = OTHER # now I can always refer to these variables

    count = 4
    finish = len(OTHER[0])
    left = count - finish % count - 1
    # print "step", step
    # print "stepsahead", count - step % count -1
    #apply km filter to decrease noise
    x = matrix([[target_measurement[0]], [target_measurement[1]]])
    new_x, P = kfilter(measurements[-25:], x, P)
    fx = new_x.value[0][0]
    fy = new_x.value[1][0]
    xy = (fx, fy)
    measurements.append(target_measurement)
    # print "m", measurements

    if len(OTHER[3]) < 5:
        OTHER[3].append(target_measurement[0])
        OTHER[4].append(target_measurement[1])
        estimate  =target_measurement
        # R = sqrt(target_measurement[0]**2 + target_measurement[1]**2)
        # n_beta = atan2(target_measurement[1], target_measurement[0])
        # dce = [0, 0]
    else:
        OTHER[3].append(target_measurement[0])
        OTHER[4].append(target_measurement[1])

        _x_ = sum(OTHER[3]) / len(OTHER[3])
        _y_ = sum(OTHER[4]) / len(OTHER[4])
        # print "other3", OTHER[3]
        a, b = [], []
        for i in range(len(OTHER[3])):
            a.append(OTHER[3][i] - _x_)
            b.append(OTHER[4][i] - _y_)
        #     # print "i", i
            # print "a", a
            # print "b", b


        dic = {"a":a,"b":b}

        # print dic.values()

        ab, aa, bb, aab, abb, aaa, bbb = [], [], [], [], [], [], []

        # t = "aab"
        # t1 = list(t)

        itemlist = ['ab', 'aa', 'bb', 'aab', 'abb', 'aaa', 'bbb']
        varlist = [ab, aa, bb, aab, abb, aaa, bbb]

        # itemlist = ['aab', 'abb']
        # varlist = [aab, abb]


        for k in range(len(itemlist)):
            t1 = list(itemlist[k])
            for i in range(len(OTHER[3])):
                x = 1
                for key in t1:
                    x *= dic[key][i]
                # exec "%s.append(%s)" % (itemlist[k], x)
            # print "aab", aab
                varlist[k].append(x)

        # print "aab", aab
        # print "abb", abb



        # for i in range(len(OTHER[3])):
        #     x = 1
        #     for key in t1:
        #         x *= dic[key][i]
        #         print "x", x
        #     aab.append(x)
        # print "aab", aab

        # for item in t1:
        #     globals().update({item: return_a_list()})
        #
        # for item in t1:
        #     print eval(item)
        #
        # print  "t1", newl
        # fd  = [aa, bb]
        # for i in fd:
        #     print "i", i
        #     for j in i:
        #         print 'fd', j
        # aab2 = []
        # abb2 = []
        # for i in range(len(OTHER[3])):
        #     a.append(OTHER[3][i] - _x_)
        #     b.append(OTHER[4][i] - _y_)
        #     ab.append(a[i]*b[i])
        #     aa.append(a[i]*a[i])
        #     bb.append(b[i]*b[i])
        #     aab2.append(a[i]*a[i]*b[i])
        #     abb2.append(a[i]*b[i]*b[i])
        #     aaa.append(a[i]*a[i]*a[i])
        #     bbb.append(b[i]*b[i]*b[i])
        # print "aab2", aab2
        # print "abb2", abb2
            # print "suu", aa

        # A = matrix([[sum(aa), sum(ab)], [sum(ab), sum(bb)]])
        # B = matrix([[sum(aaa) + sum(abb)], [sum(bbb) + sum(aab)]])
        M = matrix([[sum(aa), sum(ab)], [sum(ab), sum(bb)]]).inverse() * matrix([[sum(aaa) + sum(abb)], [sum(bbb) + sum(aab)]])
        # C_i = A * B
        # print "A", A
        # print "B", B
        # print "M", M
        # print "C_i", C_i
        dce = [M.value[0][0] / 2.0 + _x_, M.value[1][0] / 2.0 + _y_]
        # print "dce", dce

        cir = []
        # for i in range(len(OTHER[3])):
        #     cir.append(sqrt((OTHER[3][i]-dce[0])**2 + (OTHER[4][i]-dce[1])**2))
        # R = sum(cir) / len(cir)
        # print "cir", cir
        # print "R", R

        beta = []
        m_beta = 0
        degree = []
        for i in range(len(OTHER[3])):

            beta.append(atan2(OTHER[4][i] - dce[1], OTHER[3][i] - dce[0]))
            # print "beta", beta
            degree.append(degrees(beta[i]))
            # print "beta degree", degree
            m_beta = m_beta + (angle_trunc(beta[i] - beta[i - 1]) if i > 0 else 0)
            cir.append(sqrt((OTHER[3][i]-dce[0])**2 + (OTHER[4][i]-dce[1])**2))
            # off = angle_trunc(beta[i] - beta[i - 1]) if i > 0 else 0
            # m_beta += off
            # # if i > 0:
            #     # diff = beta[i] - beta[i - 1]
            #     # if diff > pi:
            #     #     diff -= 2 * pi
            #     # elif diff < -pi:
            #     #     diff += 2 * pi
            #
            #     off = angle_trunc(beta[i] - beta[i - 1])
            #     m_beta += off
        m_beta /= (len(beta) - 1)
        R = sum(cir) / len(cir)

        # print "m_", m_beta
        m__degree = degrees(m_beta)
        # print "m__degree", m__degree

        s_beta = 0
        for i in range(len(beta)):
            # s_ = (beta[i] - m_beta * i) % pi
            # s_ = s_ if beta[0] >= 0 else s_ - pi
            s_beta = s_beta + ((beta[i] - m_beta * i) % pi - (0 if beta[0] >= 0 else pi))
            # s_ = (beta[i] - m_beta * i) % pi - (0 if beta[0] >= 0 else pi)
            # if beta[0] < 0:
            #      s_ -= pi
            # s_beta += s_
            # print "s_", s_
            # s_deg = degrees(s_)
            # print "s__deg", s__deg
            # s_beta_deg = degrees(s_beta)
            # print "s_beta_deg", s_beta_deg
        s_beta /= len(beta)

        # print "s_beta", s_beta

#        n_beta = s_beta + len(beta) * m_beta
        n_beta = s_beta + (len(beta) + left) * m_beta

        # print "n_beta", n_beta
        # print "n_beta2", n_beta2
        # print "th", len(beta)
        # print "stepsahead", count
        # print "m_beta", m_beta

        estimate  = (dce[0] + R * cos(n_beta), dce[1] + R * sin(n_beta))
        # print "xy_", xy_estimate
        # print "xy_2", xy_estimate2


#        # print "i, dce, R, m_beta, s_beta: ", i, dce, R, m_beta, s_beta
    OTHER[0] = measurements
    OTHER[1].append(hunter_position)
    OTHER[2].append(hunter_heading)
    OTHER[5] = P

    # print "estimate ", estimate 
    # print "targetm", target_measurement
    # print "hunter_pos", hunter_position
    heading_to_target = get_heading(hunter_position, estimate )
    # print "head tar", heading_to_target
    turning = angle_trunc(heading_to_target - hunter_heading)
    distance = sqrt((estimate [0] - hunter_position[0])**2 + (estimate [1] - hunter_position[1])**2)

    # print "turning", turning
    # print "distance", distance
    # print "program end"
    return turning, distance, OTHER



def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
#     """Returns True if your next_move_fcn successfully guides the hunter_bot
#     to the target_bot. This function is here to help you understand how we
#     will grade your submission."""
#     max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
#     separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
#     caught = False
#     ctr = 0
#
#     # We will use your next_move_fcn until we catch the target or time expires.
#     while not caught and ctr < 1000:
#
#         # Check to see if the hunter has caught the target.
#         hunter_position = (hunter_bot.x, hunter_bot.y)
#         target_position = (target_bot.x, target_bot.y)
#         separation = distance_between(hunter_position, target_position)
#         if separation < separation_tolerance:
#             print "You got it right! It took you ", ctr, " steps to catch the target."
#             caught = True
#
#         # The target broadcasts its noisy measurement
#         target_measurement = target_bot.sense()
#
#         # This is where YOUR function will be called.
#         turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)
#
#         # Don't try to move faster than allowed!
#         if distance > max_distance:
#             distance = max_distance
#
#         # We move the hunter according to your instructions
#         hunter_bot.move(turning, distance)
#
#         # The target continues its (nearly) circular motion.
#         target_bot.move_in_circle()
#
#         ctr += 1
#         if ctr >= 1000:
#             print "It took too many steps to catch the target."
#     return caught

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.98 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization
    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        # print "hunter", hunter_position

        target_position = (target_bot.x, target_bot.y)
        # print "target", target_position
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
        #Visualize it
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught

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

target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 0.05 *target.distance
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

# print demo_grading(hunter, target, next_move)





