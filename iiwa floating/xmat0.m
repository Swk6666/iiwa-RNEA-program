function X = xmat0(theta)
    X = [cos(theta), sin(theta), 0, 0, 0, 0;
         -sin(theta), cos(theta), 0, 0, 0, 0;
         0, 0, 1, 0, 0, 0;
         -0.1575 * sin(theta), 0.1575 * cos(theta), 0, cos(theta), sin(theta), 0;
         -0.1575 * cos(theta), -0.1575 * sin(theta), 0, -sin(theta), cos(theta), 0;
         0, 0, 0, 0, 0, 1];
end
