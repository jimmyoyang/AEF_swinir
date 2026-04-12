import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 生成模拟数据 (模拟高度集中在0，带有极值的长尾分布)
# ==========================================
np.random.seed(42)
# 生成大量集中在 0 附近的数据
core_data = np.random.laplace(loc=0, scale=1.0, size=10000)
# 强制加入指定的最小值和最大值，以匹配图像中的 -18 和 43 左右的极值
min_val = -18
max_val = 43
data = np.append(core_data, [min_val, max_val])

# ==========================================
# 2. 设置图表样式 (还原干净的浅灰蓝色背景和网格)
# ==========================================
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

# 设置背景颜色类似图片中的浅色玻璃感背景
background_color = "#f4f6f9"
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# 修改网格线样式，使其更柔和
ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.5)

# ==========================================
# 3. 绘制平滑的密度曲线 (KDE) 及其填充
# ==========================================
# 使用 seaborn 的 kdeplot 画出平滑的曲线并向下填充
sns.kdeplot(
    data, 
    fill=True, 
    color="#348abd", # 沉稳的蓝色
    alpha=0.3,       # 填充透明度
    linewidth=2,     # 曲线边缘的粗细
    ax=ax,
    clip=(min_val - 2, max_val + 2) # 限制绘制范围
)

# ==========================================
# 4. 添加特殊标记 (零点线、最小值、最大值)
# ==========================================
# 绘制零点参考线
ax.axvline(0, color='gray', linestyle='-', linewidth=1.2, label='Zero', alpha=0.8)

# 绘制最小值标记 (向下的蓝色箭头/倒三角)
ax.scatter(
    min_val, 0.02, # 稍微抬高一点使其在视觉上更好看
    marker='v', color='blue', s=200, 
    edgecolor='black', linewidth=1, label='Min', zorder=5
)

# 绘制最大值标记 (向上的红色箭头/正三角)
ax.scatter(
    max_val, 0.02, 
    marker='^', color='red', s=200, 
    edgecolor='black', linewidth=1, label='Max', zorder=5
)

# ==========================================
# 5. 坐标轴与图例美化
# ==========================================
# 隐藏上边和右边的图表边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# 设置轴标签
ax.set_xlabel("Value", fontsize=14, labelpad=10, color='#333333')
ax.set_ylabel("Density", fontsize=14, labelpad=10, color='#333333')

# 设置刻度字体
ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')

# 限制 X 轴的显示范围，使其紧凑
ax.set_xlim(-20, 48)

# 设置图例：带圆角边框、阴影，放置在右上角
legend = ax.legend(
    fontsize=12, 
    loc='upper right', 
    frameon=True, 
    shadow=True, 
    fancybox=True,
    borderpad=0.8,
    labelspacing=0.8
)
legend.get_frame().set_facecolor(background_color)
legend.get_frame().set_edgecolor('gray')

# ==========================================
# 6. 显示图表
# ==========================================
plt.tight_layout()
plt.savefig("draw.png", dpi=300, bbox_inches="tight")