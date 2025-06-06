function X = xmat6(theta)
    X = [-cos(theta), 0, sin(theta), 0, 0, 0;
         sin(theta), 0, cos(theta), 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0.081 * sin(theta), 0, 0.081 * cos(theta), -cos(theta), 0, sin(theta);
         0.081 * cos(theta), 0, -0.081 * sin(theta), sin(theta), 0, cos(theta);
         0, 0, 0, 0, 1, 0];
end
