function X = xmat3(theta)
    X = [cos(theta), 0, sin(theta), 0, 0, 0;
         -sin(theta), 0, cos(theta), 0, 0, 0;
         0, -1, 0, 0, 0, 0;
         0, 0.2155 * cos(theta), 0, cos(theta), 0, sin(theta);
         0, -0.2155 * sin(theta), 0, -sin(theta), 0, cos(theta);
         0.2155, 0, 0, 0, -1, 0];
end
