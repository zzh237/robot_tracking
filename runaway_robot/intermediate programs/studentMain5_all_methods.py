# ----------
# Part Five
#
# This time, the sensor measurements from the runaway Traxbot will be VERY
# noisy (about twice the target's stepsize). You will use this noisy stream
# of measurements to localize and catch the target.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time.
#
# ----------
# GRADING
#
# Same as part 3 and 4. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
import random
import numpy as np
from numpy import linalg, array, empty, newaxis, r_, emath
from matrix import *
from scipy import optimize
from scipy import odr

def next_position_in_circle(x, y, heading, distance):
    est_x = x + distance * cos(heading)
    est_y = y + distance * sin(heading)
    return est_x, est_y


def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class KalmanFilter:
    # u = matrix([[0.], [0.], [0.]])  # external motion
    def __init__(self, sigma):
        self.x = matrix([[0.],
                         [0.],
                         [0.]])
        self.P = matrix([[1000., 0., 0.],
                         [0., 1000., 0.],
                         [0., 0., 1000.]])
        # measurement uncertainty
        self.R = matrix([[sigma, 0.],
                        [0., sigma]])
        # next state function
        self.F = matrix([[1., 1., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        # measurement function
        self.H = matrix([[1., 0., 0.],
                         [0., 0., 1.]])
        # identity matrix
        self.I = matrix([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        self.keep = []

    def predict(self, measurement):
        self.keep.append(measurement)
        # calculate heading and distance from previous data
        if len(self.keep) == 1:
            m_heading = 0
            m_distance = 0
        else:
            p1 = (self.keep[-1][0], self.keep[-1][1])
            p2 = (self.keep[-2][0], self.keep[-2][1])
            m_distance = distance_between(p1, p2)
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            m_heading = atan2(dy, dx) % (2 * pi)
            self.keep.pop(0)

        pre_heading = self.x.value[0][0]
        for d in [-1, 0, 1]:
            diff = (int(pre_heading / (2 * pi)) + d) * (2 * pi)
            if abs(m_heading + diff - pre_heading) < pi:
                m_heading += diff
                break
        # measurement update
        y = matrix([[m_heading],
                    [m_distance]]) - self.H * self.x
        S = self.H * self.P * self.H.transpose() + self.R
        K = self.P * self.H.transpose() * S.inverse()
        self.x = self.x + (K * y)
        self.P = (self.I - K * self.H) * self.P
        # prediction
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F.transpose()

        est_heading = self.x.value[0][0]
        est_distance = self.x.value[2][0]
        est_next = next_position_in_circle(measurement[0], measurement[1],
                                           est_heading, est_distance)

        return est_next


class ExtendedKalmanFilter:
    # u = matrix([[0.], [0.], [0.]])  # external motion
    def __init__(self, sigma):
        # State matrix
        self.x = matrix([[0.],      # x
                         [0.],      # y
                         [0.],      # heading
                         [0.],      # turning
                         [0.]])     # distance
        # Covariance matrix
        self.P = matrix([[1000., 0., 0., 0., 0.],
                         [0., 1000., 0., 0., 0.],
                         [0., 0., 1000., 0., 0.],
                         [0., 0., 0., 1000., 0.],
                         [0., 0., 0., 0., 1000.]])
        # measurement uncertainty
        self.R = matrix([[sigma, 0.],
                         [0., sigma]])

        # transition matrix
        self.F = ExtendedKalmanFilter.transitionMatrix(self.x)

        # measurement function
        self.H = matrix([[1., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0.]])
        # identity matrix
        self.I = matrix([[1., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0.],
                         [0., 0., 1., 0., 0.],
                         [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 1.]])
        self.keep = []



    # def kfilter(measurements, x, P):
    #     for n in range(len(measurements)):
    #
    #         # prediction
    #         x = (F * x) + a
    #         P = F * P * F.transpose()
    #
    #         # measurement update
    #         Z = matrix([measurements[n]])
    #         y = Z.transpose() - (H * x)
    #         S = H * P * H.transpose() + R
    #         K = P * H.transpose() * S.inverse()
    #         x = x + (K * y)
    #         P = (I - (K * H)) * P
    #     return x, P

    def predict(self, measurements):
        for n in range(len(measurements)):
            z = matrix([[measurements[n][0]],
                    [measurements[n][1]]])
            # z = matrix([measurements[n]])
            # Measurement update
            y = z - self.H * self.x
            S = self.H * self.P * self.H.transpose() + self.R
            K = self.P * self.H.transpose() * S.inverse()
            self.x = self.x + (K * y)
            self.P = (self.I - K * self.H) * self.P

            # Predication update
            self.F = ExtendedKalmanFilter.transitionMatrix(self.x)
            self.x = ExtendedKalmanFilter.transitionFunc(self.x)
            self.P = self.F * self.P * self.F.transpose()

        est_x = self.x.value[0][0]
        est_y = self.x.value[1][0]

        return est_x, est_y
    
    # def predict(self, measurements, x):
    #     for n in range(len(measurements)):
    #         z = matrix([[measurements[n][0]],
    #                 [measurements[n][1]]])
    #         # z = matrix([measurements[n]])
    #         # Measurement update
    #         y = z - self.H * x
    #         S = self.H * self.P * self.H.transpose() + self.R
    #         K = self.P * self.H.transpose() * S.inverse()
    #         x = x + (K * y)
    #         self.P = (self.I - K * self.H) * self.P
    #
    #         # Predication update
    #         self.F = ExtendedKalmanFilter.transitionMatrix(x)
    #         x = ExtendedKalmanFilter.transitionFunc(x)
    #         self.P = self.F * self.P * self.F.transpose()
    #
    #     est_x = x.value[0][0]
    #     est_y = x.value[1][0]
    #
    #     return est_x, est_y

    @staticmethod
    def transitionMatrix(state):
        h = state.value[2][0]
        r = state.value[3][0]
        d = state.value[4][0]
        return matrix([[1., 0., -d*sin(h+r), -d*sin(h+r), cos(h+r)],
                       [0., 1.,  d*cos(h+r),  d*cos(h+r), sin(h+r)],
                       [0., 0.,          1.,          1.,       0.],
                       [0., 0.,          0.,          1.,       0.],
                       [0., 0.,          0.,          0.,       1.]])

    @staticmethod
    def transitionFunc(state):
        x = state.value[0][0]
        y = state.value[1][0]
        h = state.value[2][0]
        r = state.value[3][0]
        d = state.value[4][0]

        x += d * cos(h + r)
        y += d * sin(h + r)
        h += r

        return matrix([[x],
                       [y],
                       [h],
                       [r],
                       [d]])


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
    
#
# def kfilter(measurements, x, P):
#     for n in range(len(measurements)):
#
#         # prediction
#         x = (F * x) + a
#         P = F * P * F.transpose()
#
#         # measurement update
#         Z = matrix([measurements[n]])
#         y = Z.transpose() - (H * x)
#         S = H * P * H.transpose() + R
#         K = P * H.transpose() * S.inverse()
#         x = x + (K * y)
#         P = (I - (K * H)) * P
#     return x, P
#
# initial_xy = [4., 12.]
# dt = 0.1
# x = matrix([[0], [0]]) # initial state (location and velocity)
# a = matrix([[0.], [0.]]) # external motion
#
# F =  matrix([[1.0, 0.0], [0.0, 1.0]])
# H =  matrix([[1.0, 0.0], [0.0, 1.0]])
# R =  matrix([[0.1, 0.0], [0.0, 0.1]])
# I =  matrix([[1.0, 0.0], [0.0, 1.0]])


def parabolic_path(x0,y0):
    return (x0+1,
            y0+2*x0)

def parabolic_ekf(x,P,meas,var_meas=0,var_move=0):
    F = matrix([[1.,0.],
                   [2.,1.]]) # the matrix of partial derivatives
                             # of the motion model
                             # f(x,y) = (x+1,2x+y)
                             # dfx/dx = 1, dfx/dy = 0
                             # dfy/dx = 2, dfy/dy = 1
    H = matrix([[1.,0.],
                   [0.,1.]]) # the matrix of partial derivatives
                             # of the measurement model
                             # h(x,y) = (x,y)
                             # dhx/dx = 1, dhx/dy = 0
                             # dhy/dx = 0, dhy/dy = 1
    Q = matrix([[var_move,0.],
                   [0.,var_move]]) # movement covariance matrix
    R = matrix([[var_meas, 0.],
                   [ 0.,var_meas]]) # measurement covariance matrix
    I = matrix([[1.,0.],
                   [0.,1.]]) # identity matrix

    # measurement update
    z = matrix([[meas[0]],
                   [meas[1]]]) # the measurement we receive

    # if we had a different function to measure than just taking
    # h(x) = x, then this would be y = z - h(x)
    y = z - x

    # the rest of the equations are the same as in the regular KF!
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - K * H) * P

    # the only real change is in the prediction update equations
    # instead of using a matrix to update the x vector, we use
    # the function f
    next_pt = parabolic_path(meas[0],meas[1])
    x = matrix([[next_pt[0]],
                   [next_pt[1]]])

    # this equation is the same as in the regular KF!
    P = F * P * F.transpose() + Q

    return x,P

def estimate_next_pos(target_meas,OTHER=None):
    if not OTHER:
        OTHER = {}
        #OTHER['meas'] = [target_meas]
        OTHER['x'] = np.matrix([[0.],
                                [0.]])
        OTHER['P'] = np.matrix([[1000.,    0.],
                                [   0., 1000.]])
    #else:
    #    OTHER['meas'].append(target_meas)

    OTHER['x'],OTHER['P'] = parabolic_ekf(OTHER['x'],OTHER['P'],target_meas,var_meas=0.2)
    x,y = np.transpose(OTHER['x']).tolist()[0]
    return (x,y), OTHER
# from numpy import *

class Circle_3b:
    
    def __init__(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.x = [x1, y1]


    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x1-xc)**2 + (self.y1-yc)**2)


    def f_3b(self, beta, x):
        """ implicit definition of the circle """
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

    def jacb(self, beta, x):
        """ Jacobian function with respect to the parameters beta.
        return df_3b/dbeta
        """
        xc, yc, r = beta
        xi, yi    = x

        df_db    = empty((beta.size, x.shape[1]))
        df_db[0] =  2*(xc-xi)                     # d_f/dxc
        df_db[1] =  2*(yc-yi)                     # d_f/dyc
        df_db[2] = -2*r                           # d_f/dr

        return df_db

    def jacd(self, beta, x):
        """ Jacobian function with respect to the input x.
        return df_3b/dx
        """
        xc, yc, r = beta
        xi, yi    = x

        df_dx    = empty_like(x)
        df_dx[0] =  2*(xi-xc)                     # d_f/dxi
        df_dx[1] =  2*(yi-yc)                     # d_f/dyi

        return df_dx

    def calc_estimate(self, data):
        """ Return a first estimation on the parameter from the data  """
        xc0, yc0 = data.x.mean(axis=1)
        r0 = sqrt((data.x[0]-xc0)**2 +(data.x[1] -yc0)**2).mean()
        return xc0, yc0, r0

    # for implicit function :
    #       data.x contains both coordinates of the points
    #       data.y is the dimensionality of the response
    def implement(self):
        lsc_data  = odr.Data(row_stack([self.x1, self.y1]), y=1)
        lsc_model = odr.Model(self.f_3b, implicit=True, estimate=self.calc_estimate, fjacd=self.jacd, fjacb=self.jacb)
        lsc_odr   = odr.ODR(lsc_data, lsc_model)    # beta0 has been replaced by an estimate function
        lsc_odr.set_job(deriv=3)                    # use user derivatives function without checking
        # lsc_odr.set_iprint(iter=1, iter_step=1)     # print details for each iteration
        lsc_out   = lsc_odr.run()
        xc_3b, yc_3b, Radius = lsc_out.beta
        Ri_3b       = self.calc_R(xc_3b, yc_3b)
        residu   = sum((Ri_3b - Radius)**2)
        dce = [xc_3b, yc_3b]
        return dce, Radius, residu
    

class Circle_3:
    def __init__(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.x = [x1, y1]
        
    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x1-xc)**2 + (self.y1-yc)**2)

    def f_3(self, beta, x):
        """ implicit definition of the circle """
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2
    
    def implement(self):
        _x_ = mean(self.x1)
        _y_ = mean(self.y1)
        # initial guess for parameters
        R_m = self.calc_R(_x_, _y_).mean()
        beta0 = [ _x_, _y_, R_m]

        # for implicit function :
        #       data.x contains both coordinates of the points (data.x = [x, y])
        #       data.y is the dimensionality of the response
        lsc_data  = odr.Data(row_stack([self.x1, self.y1]), y=1)
        lsc_model = odr.Model(self.f_3, implicit=True)
        lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
        lsc_out   = lsc_odr.run()

        xc_3, yc_3, R_3 = lsc_out.beta
        Ri_3 = self.calc_R(xc_3, yc_3)
        residu = sum((Ri_3 - R_3)**2)
        dce = [xc_3, yc_3]
        Radius = R_3
        return dce, Radius, residu

class Circle_2:
    def __init__(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.x = [x1, y1]

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x1-xc)**2 + (self.y1-yc)**2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    def implement(self):
        _x_ = mean(self.x1)
        _y_ = mean(self.y1)
        center_estimate = _x_, _y_
        center_2, ier = optimize.leastsq(self.f_2, center_estimate)

        xc_2, yc_2 = center_2
        dce = [xc_2, yc_2]
        Ri_2       = self.calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = sum((Ri_2 - R_2)**2)
        return dce, R_2, residu_2

    
class Circle_2b:
    def __init__(self, x1, y1):
        self.x1 = x1
        self.y1 = y1
        self.x = [x1, y1]  
        
    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x1-xc)**2 + (self.y1-yc)**2)
    
    def f_2b(self, c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()
    
    def Df_2b(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = empty((len(c), self.x1.size))
        Ri = self.calc_R(xc, yc)
        df2b_dc[0] = (xc - self.x1)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - self.y1)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, newaxis]
        return df2b_dc
    
    def implement(self):
        _x_ = mean(self.x1)
        _y_ = mean(self.y1)
    
        center_estimate = _x_, _y_
        center_2b, ier = optimize.leastsq(self.f_2b, center_estimate, Dfun=self.Df_2b, col_deriv=True)
        # print "center_2b", center_2b
        
        xc_2b, yc_2b = center_2b
        Ri_2b        = self.calc_R(*center_2b)
        R_2b         = Ri_2b.mean()
        residu_2b    = sum((Ri_2b - R_2b)**2)
        
        dce = [xc_2b, yc_2b]
        Radius = R_2b
        return dce, Radius, residu_2b

class Circle_1:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x1-xc)**2 + (self.y1-yc)**2)

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
        return dce


def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves.

    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = []
        hunter_positions = []
        hunter_headings = []
        xs = []
        ys = []
        P = matrix([[1000.0, 0.0], [0.0, 1000.0]])
        # OTHER1 = {}
        new_x = matrix([[0.],
                           [0.]])
        OTHER = [measurements, hunter_positions, hunter_headings, xs, ys, P, new_x] # now I can keep track of history
    else: # not the first time, update my history
        measurements, hunter_positions, hunter_headings, xs, ys, P, new_x = OTHER # now I can always refer to these variables
    # (fx,fy), OTHER1 = estimate_next_pos(target_measurement,OTHER1)
    # print "fx, fy", fx, fy
    #apply kfilter to decrease noise
    #code to use the new kfilter class
    # x = matrix([[target_measurement[0]], [target_measurement[1]]])
    measurements.append(target_measurement)
    hunter_positions.append(hunter_position)
    hunter_headings.append(hunter_heading)
    # filter = KalmanFilter(0.1)
    # fx, fy = filter.predict(target_measurement)
    # print "fx, fy", fx, fy
    #
    # # new_x, P = kfilter(measurements[-20:], x, P)
    # # fx = new_x.value[0][0]
    # # fy = new_x.value[1][0]
    # xy = (fx, fy)
    # #
    # # measurements[-1] = (xy)

    #code to use the original kfilter
    # measurements.append(target_measurement)
    # x = matrix([[target_measurement[0]], [target_measurement[1]]]) # initial state (location and velocity)
    # # x = matrix([[target_measurement[0]], [target_measurement[1]]])
    # kfilter = KFilter()
    # x, P = kfilter.filter(measurements[-30:], x, P)
    # fx = x.value[0][0]
    # fy = x.value[1][0]
    # xy = (fx, fy)
    # print "fx, fy", fx, fy

    # code to use the original kfilter
    # OTHER['x'],OTHER['P'] = parabolic_ekf(OTHER['x'],OTHER['P'],target_meas,var_meas=0.2)
    # x,y = np.transpose(OTHER['x']).tolist()[0]
    # x = matrix([[target_measurement[0]], [target_measurement[1]]])
    # new_x, P = parabolic_ekf(new_x, P, target_measurement,var_meas=0.2)
    # fx, fy = np.transpose(new_x).tolist()[0]
    # fx = new_x.value[0][0]
    # fy = new_x.value[1][0]
    # xy = (fx, fy)
    # print "fx, fy", fx, fy

    # the algorithm was referenced from http://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    # different methods used to find the least squares circle fitting a set of 2D points (x,y).
    if len(xs) < 6:
        xs.append(target_measurement[0])
        ys.append(target_measurement[1])
        pred_position  = target_measurement
    else:
        xs.append(target_measurement[0])
        ys.append(target_measurement[1])

        _x_ = sum(xs) / len(xs)
        _y_ = sum(ys) / len(ys)

        x1 = r_[xs]
        y1 = r_[ys]

        #method 1
        # a, b = [], []
        # for i in range(len(xs)):
        #     a.append(xs[i] - _x_)
        #     b.append(ys[i] - _y_)
        # dic = {"a":a,"b":b}
        # ab, aa, bb, aab, abb, aaa, bbb = [], [], [], [], [], [], []
        # itemlist = ['ab', 'aa', 'bb', 'aab', 'abb', 'aaa', 'bbb']
        # varlist = [ab, aa, bb, aab, abb, aaa, bbb]
        # for k in range(len(itemlist)):
        #     t1 = list(itemlist[k])
        #     for i in range(len(xs)):
        #         x = 1
        #         for key in t1:
        #             x *= dic[key][i]
        #         # exec "%s.append(%s)" % (itemlist[k], x)
        #     # print "aab", aab
        #         varlist[k].append(x)
        # A = array([[sum(aa), sum(ab)], [sum(ab), sum(bb)]])
        # B = array([sum(aaa) + sum(abb), sum(bbb) + sum(aab)])/2
        # uc, vc = linalg.solve(A, B)
        # xc_1 = _x_ + uc
        # yc_1 = _y_ + vc
        # dce = [xc_1, yc_1]
        # M = matrix([[sum(aa), sum(ab)], [sum(ab), sum(bb)]]).inverse() * matrix([[sum(aaa) + sum(abb)], [sum(bbb) + sum(aab)]])
        # dce = [sum(xs) / len(xs) + M.value[0][0] / 2.0, sum(ys) / len(ys) + M.value[1][0] / 2.0]
        # print "dce", dce
        # Ri = calc_R(*dce)
        # Radius = Ri.mean()
        # residu    = sum((Ri - Radius)**2)

        circle = Circle_1(xs, ys)
        # dce, Radius, residu = circle.implement()
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
        # print "r", Radius
        m_degree = degrees(m_beta)

        s_beta = 0
        count = 5
        left = count - len(OTHER[0]) % count - 1
        for i in range(len(xs)):
            s_beta = s_beta + ((beta[i] - m_beta * i) % pi - (0 if beta[0] >= 0 else pi))
        s_beta = s_beta/len(beta)
        n_beta = (len(beta) + left) * m_beta + s_beta
        pred_position  = (dce[0] + Radius * cos(n_beta), dce[1] + Radius * sin(n_beta))

        # measurements[-1] = pred_position
        # measurements.append(pred_position)

    OTHER[0] = measurements
    OTHER[1] = hunter_positions
    OTHER[2] = hunter_headings
    OTHER[3] = xs
    OTHER[4] = ys
    OTHER[5] = P
    OTHER[6] = new_x

    heading_to_target = get_heading(hunter_position, pred_position)
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
measurement_noise = 2.0*target.distance # VERY NOISY!!
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

print demo_grading(hunter, target, next_move)





