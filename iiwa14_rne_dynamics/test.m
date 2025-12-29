clc;
clear;
close all;

% 加载保存的数据
data = load('joint_angles_comparison.mat');

% 提取数据
q_desired = data.desired_joint_angle;          % 期望关节角度
qd_desired = data.desired_joint_velocity;      % 期望关节角速度
qdd_desired = data.desired_joint_acceleration; % 期望关节角加速度
tau_feedforward = data.feedforward_torque;     % 保存的前馈力矩
time = data.time;                              % 时间

% 获取数据点数量
n_samples = size(q_desired, 1);
n_joints = 7;

% 初始化MATLAB计算的力矩数组
tau_matlab = zeros(n_samples, n_joints);

% 对每个时间步使用MATLAB的RNEA计算力矩
fprintf('正在计算MATLAB RNEA力矩...\n');
for i = 1:n_samples
    tau_matlab(i, :) = rnea_torque_iiwa14(q_desired(i, :), qd_desired(i, :), qdd_desired(i, :));
end
fprintf('计算完成！\n\n');

% 计算差异
diff = tau_matlab - tau_feedforward;

% 统计信息
fprintf('========== 力矩对比统计 ==========\n');
fprintf('MATLAB vs Pinocchio 前馈力矩\n\n');
for j = 1:n_joints
    fprintf('关节 %d:\n', j);
    fprintf('  最大绝对误差: %.6e N·m\n', max(abs(diff(:, j))));
    fprintf('  均方根误差:   %.6e N·m\n', sqrt(mean(diff(:, j).^2)));
    fprintf('  平均误差:     %.6e N·m\n', mean(diff(:, j)));
    fprintf('  标准差:       %.6e N·m\n', std(diff(:, j)));
end

fprintf('\n总体最大绝对误差: %.6e N·m\n', max(max(abs(diff))));
fprintf('总体均方根误差:   %.6e N·m\n\n', sqrt(mean(mean(diff.^2))));

% 绘制对比图
figure('Position', [100, 100, 1200, 1000]);

for j = 1:n_joints
    subplot(7, 1, j);
    hold on;
    plot(time, tau_matlab(:, j), 'b-', 'LineWidth', 1.5, 'DisplayName', 'MATLAB RNEA');
    plot(time, tau_feedforward(:, j), 'r--', 'LineWidth', 1.2, 'DisplayName', 'Pinocchio (前馈)');
    hold off;
    ylabel(sprintf('Joint %d [N·m]', j));
    grid on;
    legend('Location', 'best');
    if j == 1
        title('力矩对比: MATLAB RNEA vs Pinocchio 前馈力矩');
    end
end
xlabel('Time [s]');

% 保存图像
saveas(gcf, 'matlab_vs_pinocchio_torque.fig');
saveas(gcf, 'matlab_vs_pinocchio_torque.png');
fprintf('对比图已保存到 matlab_vs_pinocchio_torque.png\n\n');

% 绘制误差图
figure('Position', [150, 150, 1200, 1000]);

for j = 1:n_joints
    subplot(7, 1, j);
    plot(time, diff(:, j), 'k-', 'LineWidth', 1);
    ylabel(sprintf('Joint %d [N·m]', j));
    grid on;
    if j == 1
        title('力矩误差 (MATLAB - Pinocchio)');
    end
end
xlabel('Time [s]');

% 保存误差图
saveas(gcf, 'torque_error.fig');
saveas(gcf, 'torque_error.png');
fprintf('误差图已保存到 torque_error.png\n');
