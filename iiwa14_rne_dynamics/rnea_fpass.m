function [v, a, f] = rnea_fpass(num_joints, parent_id_arr, Xmat_arr, S_arr, Imat_arr, qd, qdd, GRAVITY)
    n = num_joints;
    v = zeros(6,n);
    a = zeros(6,n);
    f = zeros(6,n);
    gravity_vec = zeros(6,1);
    gravity_vec(4:6) = [0; 0; -GRAVITY]; % Gravity in linear acceleration part

    for ind = 1:n
        parent_ind = parent_id_arr(ind);
        Xmat = Xmat_arr(:,:,ind);
        S = S_arr(ind,:)';

        if parent_ind == 0 % Root link
            v(:,ind) = S * qd(ind);
            % 最后这一项空间叉积项代表科里奥利加速度和离心加速度，用于计算速度矢量在旋转坐标系下的导数项
            a(:,ind) = Xmat * gravity_vec + S * qdd(ind) + cross_motion_vec(v(:,ind), S * qd(ind));
        else
            v(:,ind) = Xmat * v(:,parent_ind) + S * qd(ind);
            a(:,ind) = Xmat * a(:,parent_ind) +  S * qdd(ind) + cross_motion_vec(v(:,ind), S * qd(ind));
        end

        Imat = Imat_arr{ind};
        f(:,ind) = Imat * a(:,ind) + cross_force_vec(v(:,ind), Imat * v(:,ind));
    end
end
