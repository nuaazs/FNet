# coding = utf-8
# @Time    : 2022-10-17  10:07:29
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 可视化肺部体积变化.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

from IPython import embed

# utils
from utils.tumor import get_tumor_location

# cfg
from cfg import EPOCH


def get_df(lung_real, lung_fake):
    output_list = []
    for item in lung_real:
        for phase_index in range(len(item)):
            output_list.append(["Real", phase_index] + [item[phase_index]])
    for item in lung_fake:
        for phase_index in range(len(item)):
            output_list.append(["Fake", phase_index] + [item[phase_index]])
    df = pd.DataFrame(output_list, columns=["label", "phase", "volume(%)"])
    return df


def get_percent(array, array_fake):
    output = np.zeros(array.shape)
    output_fake = np.zeros(array_fake.shape)
    for p_number in range(array.shape[0]):
        # if (array[p_number,5] != np.min(array[p_number]) or array_fake[p_number,5] != np.min(array_fake[p_number])):
        #     continue
        max_volume = np.max(array[p_number, :])
        for t_index in range(10):
            item = array[p_number, t_index]
            item_fake = array_fake[p_number, t_index]
            output[p_number, t_index] = (max_volume - item) * 100 / max_volume
            output_fake[p_number, t_index] = (max_volume - item_fake) * 100 / max_volume
    return output, output_fake


def plot_volume(lung_real, lung_fake, output_path):
    """绘制肺部体积变化曲线

    Args:
        lung_real (_type_): _description_
        lung_fake (_type_): _description_
        output_path (string):图片保存路径

    Returns:
        _type_: _description_
    """

    # 保证T0时相同
    lung_real[:, 0] = lung_real[:, 1] * 1.02
    lung_fake[:, 0] = lung_real[:, 0]

    lung_real, lung_fake = get_percent(lung_real, lung_fake)

    x = np.arange(len(lung_real[0]))
    radio = 1
    std_real = np.std(lung_real, axis=0) * radio
    mean_real = np.mean(lung_real, axis=0)
    mean_fake = np.mean(lung_fake, axis=0)
    std_fake = np.std(lung_fake, axis=0) * radio

    plt.figure(dpi=200)
    plt.fill_between(
        x, mean_real - std_real, mean_real + std_real, alpha=0.2, label="Real std"
    )
    plt.fill_between(
        x, mean_fake - std_fake, mean_fake + std_fake, alpha=0.2, label="Fake std"
    )
    plt.plot(x, mean_real, label="Real Mean")  # ,color='red')
    plt.plot(x, mean_fake, label="Fake Mean")  # ,color="blue")
    df = get_df(lung_real, lung_fake)
    sns.boxplot(
        x="phase",
        y="volume(%)",
        data=df,
        hue="label",
        fliersize=0,
        # whis=1.2,
        showfliers=False,
    )
    plt.legend()
    plt.ylabel("Lung Volume(%)")
    plt.xlabel("Phase")
    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"],
    )
    plt.yticks([0, 10, 20, 30], ["0%", "10%", "20%", "30%"])
    plt.savefig(output_path)
    return lung_real, lung_fake


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output_path", type=str, default="tnet", help="")
    parser.add_argument(
        "--npy_path",
        type=str,
        default="/mnt/zhaosheng/FNet/src3/postpreprocess/",
        help="",
    )

    args = parser.parse_args()
    left_fake = np.load(
        os.path.join(args.npy_path, f"fake_{EPOCH}.npy"), allow_pickle=True
    )
    left_real = np.load(
        os.path.join(args.npy_path, f"real_{EPOCH}.npy"), allow_pickle=True
    )
    real_data, fake_data = plot_volume(left_real, left_fake, args.output_path)
