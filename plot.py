import re
import glob
import os
from datetime import datetime

# ⚠️ 后端必须在 import pyplot 之前设置
import matplotlib
_HEADLESS = not (os.environ.get('DISPLAY') or os.name == 'nt')
if _HEADLESS:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

# 中文字体修复（Windows 优先用微软雅黑）
for _font in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue


def parse_results(filepath):
    """解析 results 文件，返回各指标的 epoch 序列字典。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    epochs = []
    train_loss = []
    lr = []
    dice = []
    global_correct = []
    mean_iou = []
    vessel_recall = []
    vessel_precision = []
    vessel_iou = []
    bg_recall = []
    bg_precision = []
    bg_iou = []

    # 按 epoch 块分割
    blocks = re.split(r'\[epoch:\s*(\d+)\]', text)

    # blocks[0] 是第一个epoch标记之前的内容（空或training time）
    # 之后是 (epoch_num, 该epoch的数据) 交替
    for i in range(1, len(blocks), 2):
        epoch_num = int(blocks[i])
        data = blocks[i + 1]

        epochs.append(epoch_num)

        # 提取各项指标
        m = re.search(r'train_loss:\s*([\d.]+)', data)
        train_loss.append(float(m.group(1)) if m else None)

        m = re.search(r'lr:\s*([\d.e+\-]+)', data)
        lr.append(float(m.group(1)) if m else None)

        m = re.search(r'dice coefficient:\s*([\d.]+)', data)
        dice.append(float(m.group(1)) if m else None)

        m = re.search(r'global correct:\s*([\d.]+)', data)
        global_correct.append(float(m.group(1)) if m else None)

        m = re.search(r'mean IoU:\s*([\d.]+)', data)
        mean_iou.append(float(m.group(1)) if m else None)

        # 解析数组型字段
        for key, store in [('recall', [bg_recall, vessel_recall]),
                           ('precision', [bg_precision, vessel_precision]),
                           ('IoU', [bg_iou, vessel_iou])]:
            m = re.search(rf'{key}:\s*\[([^\]]+)\]', data)
            if m:
                vals = [float(v.strip().strip("'")) for v in m.group(1).split(',')]
                store[0].append(vals[0] if len(vals) > 0 else None)
                store[1].append(vals[1] if len(vals) > 1 else None)
            else:
                store[0].append(None)
                store[1].append(None)

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'lr': lr,
        'dice': dice,
        'global_correct': global_correct,
        'mean_iou': mean_iou,
        'vessel_recall': vessel_recall,
        'vessel_precision': vessel_precision,
        'vessel_iou': vessel_iou,
        'bg_recall': bg_recall,
        'bg_precision': bg_precision,
        'bg_iou': bg_iou,
    }


def plot_all(data, title=None):
    """绘制全部指标的综合仪表盘"""
    epochs = data['epochs']

    fig, axes = plt.subplots(3, 3, figsize=(22, 15))

    # 显式控制边距：top 留出 suptitle 空间，hspace/wspace 增大行列间距
    fig.subplots_adjust(
        left=0.06, right=0.96,
        bottom=0.06, top=0.92,
        hspace=0.45, wspace=0.30
    )

    fig.suptitle(title or '训练过程全景仪表盘', fontsize=20, fontweight='bold', y=0.98)

    # --- 第一行：核心训练指标 ---
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], 'b-', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, data['lr'], 'r-', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, data['dice'], 'g-', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Dice Coefficient', fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- 第二行：全局指标 ---
    ax = axes[1, 0]
    ax.plot(epochs, data['global_correct'], 'c-', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Global Accuracy', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, data['mean_iou'], 'm-', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean IoU (%)')
    ax.set_title('Mean IoU', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, data['vessel_iou'], 'orange', linewidth=1.2, label='血管 IoU')
    ax.plot(epochs, data['bg_iou'], 'gray', linewidth=1.2, alpha=0.6, label='背景 IoU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU (%)')
    ax.set_title('IoU by Class', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 第三行：血管类别详细指标 ---
    ax = axes[2, 0]
    ax.plot(epochs, data['vessel_recall'], 'orangered', linewidth=1.2, label='Vessel Recall')
    ax.plot(epochs, data['vessel_precision'], 'steelblue', linewidth=1.2, label='Vessel Precision')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('%')
    ax.set_title('Vessel Recall vs Precision', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    line1, = ax.plot(epochs, data['vessel_recall'], 'orangered', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Vessel Recall (%)', color='orangered')
    ax.tick_params(axis='y', colors='orangered')
    ax.set_title('Vessel Recall & BG Precision', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    line2, = ax2.plot(epochs, data['bg_precision'], 'purple', linewidth=1.2, alpha=0.7)
    ax2.set_ylabel('BG Precision (%)', color='purple')
    ax2.tick_params(axis='y', colors='purple')
    ax.legend([line1, line2], ['Vessel Recall', 'BG Precision'], loc='lower right', fontsize=9)

    ax = axes[2, 2]
    ax.plot(epochs, data['dice'], 'g-', linewidth=1.5, label='Dice')
    window = max(1, len(epochs) // 40)
    if window > 1:
        smoothed = []
        for s in range(len(data['dice']) - window + 1):
            smoothed.append(sum(data['dice'][s:s+window]) / window)
        smooth_x = epochs[window//2 : window//2 + len(smoothed)]
        ax.plot(smooth_x, smoothed, 'k--', linewidth=1, alpha=0.5, label=f'Smoothed (w={window})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Dice Trend + Smoothed', fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 有 GUI 则弹窗，无 GUI 则保存文件
    if _HEADLESS:
        save_path = os.path.splitext(title)[0] + '_dashboard.png' if title else 'training_dashboard.png'
        save_path = os.path.abspath(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n{'='*50}")
        print(f"[已保存] {save_path}")
        print(f"{'='*50}")
    else:
        try:
            fig.canvas.manager.window.state('zoomed')
        except Exception:
            try:
                fig.canvas.manager.window.showMaximized()
            except Exception:
                pass
        plt.show()


def choose_results_file():
    """列出所有 results 文件，让用户手动选择。"""
    candidates = sorted(glob.glob('results*.txt'), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("未找到任何 results*.txt 文件")

    print("\n" + "=" * 60)
    print("  可用的 results 文件 (按修改时间降序)")
    print("=" * 60)
    for i, f in enumerate(candidates):
        mtime = os.path.getmtime(f)
        size_kb = os.path.getsize(f) / 1024
        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        marker = " ← 最新" if i == 0 else ""
        print(f"  [{i}] {f}  ({size_kb:.1f} KB, {time_str}){marker}")

    print("-" * 60)
    choice = input("  输入编号选择文件 (直接回车选最新): ").strip()

    if choice == "":
        return candidates[0]
    try:
        idx = int(choice)
        if 0 <= idx < len(candidates):
            return candidates[idx]
        print(f"  编号 {idx} 超出范围，自动使用最新文件")
        return candidates[0]
    except ValueError:
        # 当作文件路径处理
        if os.path.exists(choice):
            return choice
        print(f"  无效输入，自动使用最新文件")
        return candidates[0]


if __name__ == '__main__':
    filepath = choose_results_file()
    print(f"\n[读取] {filepath}")

    data = parse_results(filepath)
    print(f"[解析完成] 共 {len(data['epochs'])} 个 epoch")
    print(f"  Dice 最终值:  {data['dice'][-1]:.3f}  (最佳: {max(data['dice']):.3f} @ epoch {data['epochs'][data['dice'].index(max(data['dice']))]})")
    print(f"  Loss 最终值:  {data['train_loss'][-1]:.4f}")
    print(f"  血管 Recall 最终值: {data['vessel_recall'][-1]:.1f}%")

    plot_all(data, title=os.path.basename(filepath))
