function c =rnea_bpass_floating(num_joints, Xmat_arr, f);
    n = num_joints;
    c = zeros(n,1);
    S = [0 0 1 0 0 0];
    for ind = n:-1:1
        c(ind) = S * f(:,ind+1);
        f(:,ind) = f(:,ind) + Xmat_arr(:,:,ind)'*f(:,ind+1);
        % for ind = n:-1:1
        %     S = S_arr(ind,:);
        %     c(ind) = S* f(:,ind);
        %
        %     parent_ind = parent_id_arr(ind);
    end
end
