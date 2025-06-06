from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[-cos(theta), 0, sin(theta), 0, 0, 0], [sin(theta), 0, cos(theta), 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [0.2045*sin(theta), 0, 0.2045*cos(theta), -cos(theta), 0, sin(theta)], [0.2045*cos(theta), 0, -0.2045*sin(theta), sin(theta), 0, cos(theta)], [0, 0, 0, 0, 1.0, 0]])

