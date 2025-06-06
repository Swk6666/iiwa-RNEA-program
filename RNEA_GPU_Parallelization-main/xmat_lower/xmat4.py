from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[cos(theta), sin(theta), 0, 0, 0, 0], [-sin(theta), cos(theta), 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0], [-0.21884625*sin(theta), 0.21884625*cos(theta), 2.5725*cos(theta), cos(theta), sin(theta), 0], [-0.21884625*cos(theta), -0.21884625*sin(theta), -2.5725*sin(theta), -sin(theta), cos(theta), 0], [-2.5725, 0, 0, 0, 0, 1.0]])

