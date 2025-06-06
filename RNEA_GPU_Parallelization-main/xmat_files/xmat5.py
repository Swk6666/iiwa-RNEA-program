from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[cos(theta), 0, sin(theta), 0, 0, 0], [-sin(theta), 0, cos(theta), 0, 0, 0], [0, -1.0, 0, 0, 0, 0], [0, 0.2155*cos(theta), 0, cos(theta), 0, sin(theta)], [0, -0.2155*sin(theta), 0, -sin(theta), 0, cos(theta)], [0.2155, 0, 0, 0, -1.0, 0]])

