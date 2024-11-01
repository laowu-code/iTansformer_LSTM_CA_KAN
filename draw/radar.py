from matplotlib import pyplot as plt
from pycirclize import Circos
import pandas as pd
from matplotlib import rc

# 设置全局字体为Arial
rc('font', family='Arial')
# 创建4个不同的DataFrame（例如不同的季节数据）
df4 = pd.DataFrame(
    data=[
        [0.0006,	-0.0047,	0.0258,	0.0126,
],
        [0.0004,	-0.0062,	0.0106,	0.0032,
],
        [0.0387,	-0.0052,	0.0373,	0.0115,
],
    ],
    index=["Proposed", "iTransformer", "LSTM"],
    columns=["Spring", "Summer", "Autumn", "Winter"]
).round(4)

df1 = pd.DataFrame(
    data=[
        [0.1335,	0.0517,	0.0986,	0.0428,
],
        [0.1455,	0.0591,	0.1073,	0.0468,
],
        [0.1448,	0.0533,	0.1182,	0.0473,
],
    ],
    index=["Proposed", "iTransformer", "LSTM"],
    columns=["Spring", "Summer", "Autumn", "Winter"]
).round(4)

df2 = pd.DataFrame(
    data=[
        [0.3406,	0.1153,	0.2574,	0.1717,
],
        [0.3448,	0.1316,	0.2805,	0.1806,
],
        [0.3542,	0.1278,	0.323,	0.1907,
],
    ],
    index=["Proposed", "iTransformer", "LSTM"],
    columns=["Spring", "Summer", "Autumn", "Winter"]
).round(4)

df3 = pd.DataFrame(
    data=[
        [0.9646,	0.9966,	0.9761,	0.9913,
],
        [0.9637,	0.9956,	0.9716,	0.9903,
],
        [0.9617,	0.9956,	0.9623,	0.9892,
],
    ],
    index=["Proposed", "iTransformer", "LSTM"],
    columns=["Spring", "Summer", "Autumn", "Winter"]
).round(4)

# 准备画布，并创建2x2的子图
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# axes = axes.flatten()  # 将子图轴平铺为一维数组，便于迭代
fig = plt.figure(figsize=(12, 12), dpi=100)
fig.subplots(2, 2, subplot_kw=dict(polar=True))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
# 列表保存每个数据集的名字，方便设置子图标题
dfs = [df1, df2, df3, df4]
titles = ['$\mathrm{MAE}$', '$\mathrm{RMSE}$', r'$\mathrm{R^2}$', r'$\mathrm{MBE}$']
v_max=[0.1548,0.3642,0.998,0.039]
v_min=[0.042,0.105,0.961,-0.007]
# 绘制4个子图
for i, (df, ax) in enumerate(zip(dfs, fig.axes)):

    print(df)
    circos = Circos.radar_chart(
        df,
        vmax=v_max[i],
        vmin=v_min[i],
        grid_interval_ratio=0.3,
        grid_line_kws=dict(lw=1, color='black', ls="--"),
        grid_label_kws=dict(size=10, color='black'),
        grid_label_formatter=lambda v: f"{v:.4f}",
        line_kws_handler=lambda _: dict(lw=1.2, ls="solid"),
          # 将子图的轴传入Circos的绘制
    )
    circos.plotfig(ax=ax)
    # 添加子图标题
    # circos.text("RPG Jobs Radar Chart", r=125, size=15, weight="bold")
    # circos.text(titles[i], r=125, size=15, weight="bold")
    # 在右上角添加图例
    ax.set_title(titles[i],size=15, weight="bold",color="black")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=10)

# 调整布局，使子图之间的间距适当
# plt.tight_layout()
plt.savefig('pic_/radar.png', dpi=800)
plt.savefig('pic_/radar.svg',format='SVG',dpi=600)
plt.show()
