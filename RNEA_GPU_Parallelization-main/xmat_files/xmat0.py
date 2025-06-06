from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[1.0*cos(theta), 1.0*sin(theta), 0, 0, 0, 0], [-1.0*sin(theta), 1.0*cos(theta), 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0], [-0.1575*sin(theta), 0.1575*cos(theta), 0, 1.0*cos(theta), 1.0*sin(theta), 0], [-0.1575*cos(theta), -0.1575*sin(theta), 0, -1.0*sin(theta), 1.0*cos(theta), 0], [0, 0, 0, 0, 0, 1.0]])

