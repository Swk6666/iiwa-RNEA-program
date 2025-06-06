% 示例数据
FontSize=12;
lablesize=14;
x = -0.5:0.001:1.5;
y_simulation = 0.5 * tan((pi / 2) * (x - 0.5));  % 仿真结果

% 创建绘图
figure; % 新建一个图窗口
set(gcf, 'Position', [100, 100, 350*1.6180339887, 350]); % 设置窗口位置和大小：[左, 下, 宽, 高]

% 绘图
plot(x, y_simulation, 'Color', [0.0, 0.4, 0.8], 'LineWidth', 2); % 深蓝色
hold off;

% 设置 y 轴范围
ylim([-3, 6]);

% 设置 x 轴范围
xlim([-0.5, 1.5]);

% 图例与样式
legend('$\varepsilon \cdot \tan\left(\frac{\pi}{2} \cdot \left(F(q) - 0.5\right)\right)$', ...
    'Interpreter', 'latex', ...   % 设置为 LaTeX 格式
    'FontSize', 14, ...           % 设置字体大小
    'Location', 'northeast');     % 设置图例位置

% 设置坐标轴标签并更改为 LaTeX 格式，并设置字体大小
xlabel('distance/F(q)', 'FontSize', lablesize, 'Interpreter', 'latex');   % x轴标签
ylabel('value ', 'FontSize', lablesize, 'Interpreter', 'latex');   % y轴标签



% 设置坐标轴刻度字体大小
ax = gca;  % 获取当前坐标轴
ax.FontSize = FontSize;  % 设置x轴和y轴的刻度字体大小

% 网格
grid on;
