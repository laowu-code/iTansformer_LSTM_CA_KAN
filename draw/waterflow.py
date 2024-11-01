import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import rc
rc('font', family='Arial')


def line_3d(x, y, z,y_label, x_label_indexs):
    """
    在y轴的每个点，向x轴的方向延伸出一个折线面：展示每个变量的时序变化。
    x: x轴，时间维，右边。
    y: y轴，变量维，左边。
    z: z轴，数值维。二维矩阵，y列x行。每一行是对应变量的一个时间序列。
    x_label_indexs: 需要标注的时间点。
    """
    x_num = len(x)
    y_num = len(y)
    if z.shape[0] != y_num or z.shape[1] != x_num:
        print('Invalid')
        return -1
    y_label=y_label


    # 制作坐标格点(z中每个点对应的x、y坐标)
    X, Y = np.meshgrid(x, y)

    # 初始化
    canvas = plt.figure(figsize=(12, 8))  # 创建画布
    axs = canvas.add_subplot(111, projection='3d')  # 添加三维子图
    # 若把111改成234，则意思是：创建一个2*3的网格，并在第4个格子中创建一个axes
    axs.set_box_aspect([2, 5, 1])
    # 绘制折线面
    for i in range(y_num):  # 遍历
        # z值线，即实际数据。
        axs.plot(Y[i], X[i], z[i], color=plt.cm.viridis(i / y_num),
                 linestyle='-', linewidth=1,  markersize=3, alpha=0.5)#marker='o',
        # 0值线（z=0），与“地面”连接。
        axs.plot(Y[i], X[i], np.zeros_like(z[i]), color='gray', alpha=0.5)


        # 绘制有颜色的平面：本质是填充z值与0值之间的区域。
        polygon = [
            [Y[i, 0], X[i, 0], 0],  # 左下
            [Y[i, -1], X[i, -1], 0],  # 右下
        ]
        for j in range(x_num - 1, -1, -1):  # 依次添加点，使得polygon成为一个完整的闭合多边形
            polygon.append([Y[i, j], X[i, j], z[i, j]])
        axs.add_collection3d(Poly3DCollection([polygon], color=plt.cm.viridis(i / y_num), alpha=0.3))

        # 标注数字（z值）
        for k in x_label_indexs:
            axs.text(Y[i, k] - 0.05, X[i, k], z[i, k] + 0.02, f'{z[i, k]:.2f}',
                     color='black', ha='center', size=7)

    # 用虚线将需要标注的时间（y）连起来
    for k in x_label_indexs:
        axs.plot(Y[:, k], X[:, k], z[:, k], linestyle='--', linewidth=0.8, color='gray')
    axs.set_xticks(y)
    axs.set_zticks(np.linspace(0,0.9,4))
    axs.set_xticklabels(y_label, rotation=0, ha='right', fontsize=10)
    axs.set_ylabel('Time(h)',labelpad=20,  fontsize=10)
    axs.set_xlabel('Variations', labelpad=10, fontsize=10)
    axs.set_zlabel('Value', labelpad=5, fontsize=10,rotation=180)
    # axs.grid(False)
    axs.view_init(elev=20, azim=-60)
    # axs.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # axs.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # axs.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # axs.grid()
    plt.savefig('pic_/data_distribution.png', dpi=400)
    plt.savefig('pic_/data_distribution.svg',format='svg', dpi=600)
    plt.show()


if __name__ == '__main__':


    site = '7-First-Solar'
    # dataset = 'Spring'
    # dataset = 'Summer'
    dataset = 'Autumn'
    # dataset = 'Winter'
    # file_path = f'./data/{site}/{dataset}_{site}_2017_2020_h.csv'
    file_path = f'./data/{site}/{dataset}_{site}_2019_2022_h.csv'
    df_all = pd.read_csv(file_path, header=0)

    z=df_all.values[:,1:]
    z[:,0]=np.maximum(0,z[:,0])


    scalar=MinMaxScaler()
    scalar.fit(z)
    z = scalar.transform(z)
    z = z.transpose(1, 0)
    z=z[::-1,:300]

    time = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    # z = np.array(
    #     [
    #         [0.20, 0.34, 0.38, 0.43, 0.44, 0.50, 0.61],
    #         [0.21, 0.40, 0.38, 0.43, 0.60, 0.72, 0.75],
    #         [0.22, 0.43, 0.44, 0.60, 0.77, 0.84, 0.92],
    #         [0.23, 0.42, 0.44, 0.43, 0.64, 0.77, 0.86],
    #         [0.38, 0.42, 0.43, 0.49, 0.55, 0.60, 0.81]
    #     ]
    # )

    y_label = ['AP', 'T', 'RH', 'GHI', 'DHI']
    y_label=y_label[::-1]
    line_3d(time, y, z,y_label, [])