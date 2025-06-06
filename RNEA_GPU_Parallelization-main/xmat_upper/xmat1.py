from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[-3.6732070158254e-6*cos(theta), sin(theta), -cos(theta), 0, 0, 0], [3.6732070158254e-6*sin(theta), cos(theta), sin(theta), 0, 0, 0], [1.0, 0, -3.6732070158254e-6, 0, 0, 0], [-0.26299*sin(theta), 0.0813080339936694*cos(theta), 0.081309*sin(theta), -3.6732070158254e-6*cos(theta), sin(theta), -cos(theta)], [-0.26299*cos(theta), -0.0813080339936694*sin(theta), 0.081309*cos(theta), 3.6732070158254e-6*sin(theta), cos(theta), sin(theta)], [0, 0.262990298663186, 0, 1.0, 0, -3.6732070158254e-6]])

