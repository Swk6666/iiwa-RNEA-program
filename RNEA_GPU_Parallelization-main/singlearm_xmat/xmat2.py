from numpy import array, sin, cos

def _lambdifygenerated(theta):
    return array([[cos(theta), -3.6732070158254e-6*sin(theta), sin(theta), 0, 0, 0], [-sin(theta), -3.6732070158254e-6*cos(theta), cos(theta), 0, 0, 0], [0, -1.0, -3.6732070158254e-6, 0, 0, 0], [-0.113519308444377*sin(theta), 0.18827*cos(theta), 0.11352*cos(theta), cos(theta), -3.6732070158254e-6*sin(theta), sin(theta)], [-0.113519308444377*cos(theta), -0.18827*sin(theta), -0.11352*sin(theta), -sin(theta), -3.6732070158254e-6*cos(theta), cos(theta)], [0.188270416980522, 0, 0, 0, -1.0, -3.6732070158254e-6]])

