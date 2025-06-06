function result = cross_motion_vec(v, m)
    % Compute the spatial motion cross product: v x m
    w = v(1:3);
    v_lin = v(4:6);
    m_w = m(1:3);
    m_v = m(4:6);
    result = zeros(6,1);
    result(1:3) = cross(w, m_w);
    result(4:6) = cross(w, m_v) + cross(v_lin, m_w);
end
