function X = xmat1(theta)
    X = [-cos(theta), 0, sin(theta), 0, 0, 0;
         sin(theta), 0, cos(theta), 0, 0, 0;
         0, 1, 0, 0, 0, 0;
         0, -0.2025 * cos(theta), 0, -cos(theta), 0, sin(theta);
         0, 0.2025 * sin(theta), 0, sin(theta), 0, cos(theta);
         -0.2025, 0, 0, 0, 1, 0];
end
