function X = xmat4(theta)
    X = [-cos(theta), 0, sin(theta), 0, 0, 0;
         sin(theta), 0, cos(theta), 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0.1845 * sin(theta), 0, 0.1845 * cos(theta), -cos(theta), 0, sin(theta);
         0.1845 * cos(theta), 0, -0.1845 * sin(theta), sin(theta), 0, cos(theta);
         0, 0, 0, 0, 1, 0];
end
