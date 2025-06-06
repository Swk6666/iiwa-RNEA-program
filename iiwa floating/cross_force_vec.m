function result = cross_force_vec(v, f)
    % Compute the spatial force cross product: v x* f
    w = v(1:3);
    v_lin = v(4:6);
    f_mom = f(1:3);
    f_frc = f(4:6);
    result = zeros(6,1);
    result(1:3) = cross(v_lin, f_frc) + cross(w, f_mom);
    result(4:6) = cross(w, f_frc);
end
