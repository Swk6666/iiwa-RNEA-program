function crossOperator = crossOperatorFunction(v)
    % 输入参数 v 是一个三维向量 [x; y; z]
    % 输出 crossOperator 是对应的叉乘操作算子

    % 提取向量分量
    x = v(1,1);
    y = v(2,1);
    z = v(3,1);

    % 创建叉乘算子
    crossOperator = [0, -z, y;
                     z, 0, -x;
                    -y, x, 0];
end