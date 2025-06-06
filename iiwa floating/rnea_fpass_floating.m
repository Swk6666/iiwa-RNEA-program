function [v, a, f] = rnea_fpass_floating(num_joints, parent_id_arr, Xmat_arr, S_arr, Imat_arr, qd, qdd, r0_d, r0_dd)
    n = num_joints+1;
    v = zeros(6,n);
    a = zeros(6,n);
    f = zeros(6,n);
    gravity_vec = zeros(6,1);
    gravity_vec(4:6) = [0; 0; 0]; % Gravity in linear acceleration part
    v(:,1) = r0_d;
    a(:,1) = r0_dd;
    S = [0 0 1 0 0 0];
    S=S';
    % Xmat_1 =[0.8776,0.4794,0;
    %     -0.4794,0.8776,0;
    %     0,0,1]; 
    % Xmat_2=zeros(3,3);
    % r = [0,0,0.1575];
    % Xmat_3=-1*Xmat_1*crossOperatorFunction(r');
    % Xmat = [Xmat_1,Xmat_2;Xmat_3,Xmat_1];

    % Xmat=Xmat_arr(:,:,1)
    % v(:,2) = Xmat * v(:,1) + S * 0.1;
    % a(:,2) = Xmat * a(:,1) + cross_motion_vec(v(:,2) , S * 0.1) + S * 0.1;

    for ind=1:7
        Xmat = Xmat_arr(:,:,ind);
        v(:,ind+1) = Xmat * v(:,ind) + S * qd(ind);
        a(:,ind+1) = Xmat * a(:,ind) + cross_motion_vec(v(:,ind+1) , S * qd(ind)) + S * qdd(ind);
        Imat = Imat_arr{ind+1};
        f(:,ind+1) = Imat * a(:,ind+1) + cross_force_vec(v(:,ind+1), Imat * v(:,ind+1));
    end


    % for ind = 1:n
    %     parent_ind = parent_id_arr(ind);
    %     Xmat = Xmat_arr(:,:,ind);
    %     S = S_arr(ind,:)';
        
        % if parent_ind == 0 % Root link
        %     v(:,ind) = r0_d;
        %     a(:,ind) = r0_dd;
        % else
        %     v(:,ind) = Xmat * v(:,parent_ind) + S * qd(ind);
        %     a(:,ind) = Xmat * a(:,parent_ind) + cross_motion_vec(v(:,ind), S * qd(ind)) + S * qdd(ind);
        % end
        % 
        % Imat = Imat_arr{ind};
        % f(:,ind) = Imat * a(:,ind) + cross_force_vec(v(:,ind), Imat * v(:,ind));
    % end
end
