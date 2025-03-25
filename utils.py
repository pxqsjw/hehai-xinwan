import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from action_opt import configs

plt.rcParams["font.family"] = ["DejaVu Sans"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def save_txt(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def save_evaluate(y_test, y_pred, output_path):
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(configs.LABEL_SETS))),
        target_names=configs.LABEL_SETS,
        digits=4,
        zero_division=0,
    )  # 计算性能指标 包括precision/recall/f1-score
    matrix = multilabel_confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
    save_txt(output_path, report + "\n\n" + str(matrix))  # 保存性能指标和混淆矩阵
    # print(report)  # 输出性能指标
    # print(f"Confusion Matrix:\n{matrix}")  # 输出混淆矩阵


def epoch_visualization(y1, y2, name, output_path):
    # 绘制epoch变化图
    plt.figure(figsize=(16, 9), dpi=100)  # 定义画布
    plt.plot(y1, marker=".", linestyle="-", linewidth=2, label=f"train {name}")  # 曲线
    plt.plot(y2, marker=".", linestyle="-", linewidth=2, label=f"val {name}")  # 曲线
    plt.title(f"训练过程中 {name} 变化图", fontsize=24)  # 标题
    plt.xlabel("epoch", fontsize=20)  # x轴标签
    plt.ylabel(name, fontsize=20)  # y轴标签
    plt.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    plt.legend(loc="best", prop={"size": 20})
    plt.savefig(output_path)  # 保存图像
    plt.show()  # 显示
