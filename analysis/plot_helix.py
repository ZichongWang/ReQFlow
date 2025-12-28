import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
plt.rc('font',family='Arial')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    # "font.family": "serif",           # Use serif font
    # "font.serif": ["Times New Roman"], # Specify Times New Roman
    "axes.titlesize": 16,            # Title font size
    "axes.labelsize": 14,            # Axis labels font size
    "xtick.labelsize": 12,           # X-axis tick font size
    "ytick.labelsize": 12,           # Y-axis tick font size
    "legend.fontsize": 12,           # Legend font size
    "legend.title_fontsize": 12,      # Legend title font size
    # "text.usetex": True
})

df = pd.read_csv("analysis/data/ReQFlow_PDB_Helix.csv")
df_new = pd.read_csv("/home/zichong_wang/ReQFlow/inference_outputs/ckpts/reqflow_pdb_rectify/reqflow_pdb_rectify/unconditional/inference_outputs/run_2025-12-27_15-53-55/Single_PDB_Metrics.csv")
df_new = df_new[['helix_percent', 'strand_percent']]
df_new.insert(0, "method", "current")

df = pd.concat([df, df_new])


df =  df[~df.select_dtypes(include=['number']).apply(lambda row: (row > 1).any() or (row < 0).any(), axis=1)]
# 2. 获取所有的 method
methods = df['method'].unique()

# 3. 先遍历所有 method，计算 2D 直方图（bins=10），
#    并做归一化(除以总和 -> frequency)
histograms = []
xedges_list = []
yedges_list = []

for m in methods:
    sub_df = df[df['method'] == m]
    H, xedges, yedges = np.histogram2d(
        sub_df['helix_percent'],
        sub_df['strand_percent'],
        bins=10,
        range=[[0, 1], [0, 1]],
    )
    H = H / H.sum()  # 归一化
    H = np.clip(H, 0, 0.15)  
    histograms.append(H)
    xedges_list.append(xedges)
    yedges_list.append(yedges)

# 4. 找到所有热力图中的最大值，用于设定统一的 vmax
max_val = max(H.max() for H in histograms)

# 5. 自定义一个新的 colormap，使最低值(0)对应白色
#    先从 Greens 取出 256 个颜色，然后将第 0 个颜色设置为白色
base_cmap = plt.cm.get_cmap('Reds', 256)       # 从 'Greens' 获取 256 级颜色
newcolors = base_cmap(np.linspace(0, 0.7, 256))    # 转成 RGBA 数组
newcolors[0, :] = np.array([1, 1, 1, 1])         # 第一个颜色改为白色(RGBA=1,1,1,1)
white_cmap = mcolors.ListedColormap(newcolors)   # 生成新的 cmap

# 6. 创建绘图区域：1 行 5 列
#    figsize 你可以自行调整，下面示例用 (20,4)；可以适当减小/增大
fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharex=True, sharey=True)

# 7. 调整子图与画布边缘的间距；这里留出右侧空间给 colorbar
#    wspace 减小以压缩子图之间的水平间距
plt.subplots_adjust(left=0.05, right=0.88, bottom=0.15, top=0.85, wspace=0.2)

# 8. 逐个子图绘制
for i, m in enumerate(methods):
    ax = axes[i]
    H = histograms[i]
    # xedges = xedges_list[i]  # 如果想在坐标轴上精确还原 x,y 范围，可以修改 extent
    # yedges = yedges_list[i]
    
    # extent=[0,1,0,1] 表示将 x,y 范围都固定到 [0,1]
    # 如果你想保留原始分箱范围，可以用: 
    #    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        H.T,
        origin='lower',
        aspect='equal',
        extent=[0,1,0,1],
        vmin=0,
        vmax=max_val,
        cmap=white_cmap  # 使用自定义的 cmap
    )
    
    ax.set_title(m, fontsize=20)
    ax.set_xlabel("Helix percent", fontsize=20)
    if i == 0:
        ax.set_ylabel("Strand percent", fontsize=20)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=18)


# 9. 在右侧手动添加 colorbar
#    先在 figure 上开辟一个坐标轴区域 [x0, y0, width, height]
#    这里可以根据实际需求微调
cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Frequency", fontsize=18)
ticks = [0.0, 0.05, 0.1, 0.15]
cbar.set_ticks(ticks)
tick_labels = [f"{val:.2f}" for val in ticks]
tick_labels[-1] = ">0.15"  # 最后一个改成 ">0.4"
cbar.set_ticklabels(tick_labels)
cbar.ax.tick_params(labelsize=18)
plt.savefig("combined_helix_strand_percent_heatmap.pdf")