function c = rnea_bpass(num_joints, parent_id_arr, Xmat_arr, S_arr, f)
    n = num_joints;
    c = zeros(n,1);

    for ind = n:-1:1
        S = S_arr(ind,:);
        c(ind) = S* f(:,ind);

        parent_ind = parent_id_arr(ind);
        if parent_ind ~= 0
            Xmat = Xmat_arr(:,:,ind);

            f(:,parent_ind) = f(:,parent_ind) + Xmat' * f(:,ind);
        end
    end
end
