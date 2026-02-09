import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import warnings

# 忽略 NaN 计算警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================

BASE_DIR = os.getcwd()
# 请确认您的相对路径
GT_REL_PATH = os.path.join("data", "cholec80", "gt-phase")
PRED_REL_PATH = os.path.join("bimask_ss_pos", "cholec80", "stage2_40_40", "output", "phase2", "Test")
SAVE_DIR = "evaluation_results"

TEST_VIDEO_IDS = range(41, 81)
FPS = 1
TOLERANCE = 10  # 10帧松弛

PHASE_MAP = {
    0: {'name': 'Preparation',               'color': '#D3D3D3'},
    1: {'name': 'CalotTriangleDissection',   'color': '#FFA500'},
    2: {'name': 'ClippingCutting',           'color': '#00FFFF'},
    3: {'name': 'GallbladderDissection',     'color': '#0000FF'},
    4: {'name': 'GallbladderPackaging',      'color': '#FF00FF'},
    5: {'name': 'CleaningCoagulation',       'color': '#008000'},
    6: {'name': 'GallbladderRetraction',     'color': '#FFFF00'}
}

# ================= 核心计算函数 (严格复刻 Evaluate.m) =================

def evaluate_strict_boundary(y_gt, y_pred, num_phases=7, tolerance=10):
    """
    严格复刻 MATLAB Evaluate.m 的条件松弛逻辑
    """
    y_gt = np.array(y_gt, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # MATLAB: diff = predLabelID - gtLabelID
    diff = y_pred - y_gt

    # 复制一份用于修正 (updatedDiff)
    updated_diff = diff.copy()

    # 遍历每个阶段 (Python 0-6 对应 MATLAB 1-7)
    for phase_id in range(num_phases):
        # 找连通域 (相当于 bwconncomp)
        # 获取该阶段的所有索引
        is_phase = (y_gt == phase_id)
        if not np.any(is_phase):
            continue

        # 寻找连通区域的起止点
        # padding 0 使得 diff 可以检测边缘
        padded = np.pad(is_phase.astype(int), (1, 1), 'constant', constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] # end是切片独占索引，刚好符合Python切片习惯

        for start, end in zip(starts, ends):
            segment_len = end - start
            t = min(tolerance, segment_len)

            # 获取当前段内的 diff 引用
            # 注意：我们需要修改的是 updated_diff，判断依据是原始 diff

            # === 严格复刻 MATLAB 的 if-elseif-else 逻辑 ===
            # MATLAB Phase 1-7 -> Python Phase 0-6

            # 头部切片 (Head Slice): start : start+t
            head_indices = slice(start, start+t)
            # 尾部切片 (Tail Slice): end-t : end
            tail_indices = slice(end-t, end)

            # 获取头部和尾部的原始误差值
            head_vals = diff[head_indices]
            tail_vals = diff[tail_indices]

            # 准备修正掩码 (True 表示需要置 0)
            head_fix_mask = np.zeros_like(head_vals, dtype=bool)
            tail_fix_mask = np.zeros_like(tail_vals, dtype=bool)

            # MATLAB: if(iPhase == 4 || iPhase == 5) -> Python 3 or 4
            if phase_id == 3 or phase_id == 4:
                # curDiff(curDiff(1:t)==-1) = 0
                head_fix_mask = (head_vals == -1)
                # curDiff(curDiff(end-t+1:end)==1 | curDiff(end-t+1:end)==2) = 0
                tail_fix_mask = (tail_vals == 1) | (tail_vals == 2)

            # MATLAB: elseif(iPhase == 6 || iPhase == 7) -> Python 5 or 6
            elif phase_id == 5 or phase_id == 6:
                # curDiff(curDiff(1:t)==-1 | curDiff(1:t)==-2) = 0
                head_fix_mask = (head_vals == -1) | (head_vals == -2)
                # curDiff(curDiff(end-t+1:end)==1 | curDiff(end-t+1:end)==2) = 0
                tail_fix_mask = (tail_vals == 1) | (tail_vals == 2)

            # MATLAB: else (Phase 1, 2, 3) -> Python 0, 1, 2
            else:
                # curDiff(curDiff(1:t)==-1) = 0
                head_fix_mask = (head_vals == -1)
                # curDiff(curDiff(end-t+1:end)==1) = 0
                tail_fix_mask = (tail_vals == 1)

            # 应用修正
            # 注意：不能直接 updated_diff[head_indices][head_fix_mask] = 0，因为切片返回副本
            # 必须使用完整的索引操作

            # 修正头部
            # 找到 head_indices 中需要修正的相对位置，加上 start 得到绝对位置
            updated_diff[start : start+t][head_fix_mask] = 0

            # 修正尾部
            updated_diff[end-t : end][tail_fix_mask] = 0

    # === 计算指标 ===
    prec_list = []
    rec_list = []
    jacc_list = []

    for phase_id in range(num_phases):
        gt_mask = (y_gt == phase_id)
        pred_mask = (y_pred == phase_id)

        # MATLAB: if(gtConn.NumObjects == 0) -> NaN
        if not np.any(gt_mask):
            prec_list.append(np.nan)
            rec_list.append(np.nan)
            jacc_list.append(np.nan)
            continue

        # union
        union_mask = gt_mask | pred_mask
        union_indices = np.where(union_mask)[0]

        # TP: updatedDiff(iPUnion) == 0
        # 在并集区域内，修正后的 diff 为 0 的数量
        tp = np.sum(updated_diff[union_indices] == 0)

        # Jaccard = TP / Union
        jacc = (tp / len(union_indices)) * 100

        # Precision = TP / Pred_Count (Relaxed)
        pred_count = np.sum(pred_mask)
        gt_count = np.sum(gt_mask)

        # 防止除零
        prec = (tp / pred_count * 100) if pred_count > 0 else 0
        rec = (tp / gt_count * 100) if gt_count > 0 else 0

        prec_list.append(prec)
        rec_list.append(rec)
        jacc_list.append(jacc)

    # Accuracy (Relaxed)
    total_correct = np.sum(updated_diff == 0)
    acc = (total_correct / len(y_gt)) * 100

    return acc, prec_list, rec_list, jacc_list

# ================= 辅助函数 =================

def read_phase_file(path):
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 2:
                labels.append(int(parts[1]))
            else:
                labels.append(int(parts[0]))
    return np.array(labels)

def plot_ribbon(gt, pred, video_name, save_path):
    """(绘图代码保持不变)"""
    phases = sorted(PHASE_MAP.keys())
    colors = [PHASE_MAP[i]['color'] for i in phases]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 3), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax1.imshow(gt[np.newaxis, :], aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(colors), vmin=0, vmax=6)
    ax1.set_ylabel('Ground Truth', fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center')
    ax1.set_yticks([])
    ax2.imshow(pred[np.newaxis, :], aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(colors), vmin=0, vmax=6)
    ax2.set_ylabel('Prediction', fontsize=12, fontweight='bold', rotation=0, labelpad=60, va='center')
    ax2.set_yticks([])
    ax2.set_xlabel('Time (Frames)', fontsize=10)
    fig.suptitle(f'Surgical Phase Recognition: {video_name}', fontsize=14, y=0.98)
    patches = [mpatches.Patch(color=PHASE_MAP[i]['color'], label=f"P{i}: {PHASE_MAP[i]['name']}") for i in phases]
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# ================= 主程序 (严格复刻 Main.m) =================

def main():
    GT_DIR = os.path.join(BASE_DIR, GT_REL_PATH)
    PRED_DIR = os.path.join(BASE_DIR, PRED_REL_PATH)
    OUT_DIR = os.path.join(BASE_DIR, SAVE_DIR)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print(f"[*] Processing videos with MICCAI STRICT Logic...")

    num_videos = len(TEST_VIDEO_IDS)
    num_phases = len(PHASE_MAP)

    # 初始化矩阵为 NaN
    mat_prec = np.full((num_videos, num_phases), np.nan)
    mat_rec  = np.full((num_videos, num_phases), np.nan)
    mat_jacc = np.full((num_videos, num_phases), np.nan)
    list_acc = []

    for idx, vid in enumerate(tqdm(TEST_VIDEO_IDS, desc="Eval")):
        video_name = f"video{vid:02d}"
        gt_file = os.path.join(GT_DIR, f"{video_name}-phase.txt")
        pred_file = os.path.join(PRED_DIR, f"{video_name}-phase.txt")

        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            continue

        y_gt = read_phase_file(gt_file)
        y_pred = read_phase_file(pred_file)

        # 对齐
        min_len = min(len(y_gt), len(y_pred))
        y_gt = y_gt[:min_len]
        y_pred = y_pred[:min_len]

        # 绘图
        plot_ribbon(y_gt, y_pred, video_name, os.path.join(OUT_DIR, f"{video_name}_vis.png"))

        # 计算严格指标
        acc, p_list, r_list, j_list = evaluate_strict_boundary(y_gt, y_pred, num_phases, TOLERANCE)

        list_acc.append(acc)

        for i in range(num_phases):
            mat_prec[idx, i] = p_list[i]
            mat_rec[idx, i]  = r_list[i]
            mat_jacc[idx, i] = j_list[i]

    # === MATLAB Main.m 的截断逻辑 (Clamping) ===
    # index = find(jaccard>100); jaccard(index)=100;
    # 这一步非常重要，因为松弛边界可能导致 Precision > 100
    mat_prec = np.clip(mat_prec, 0, 100)
    mat_rec  = np.clip(mat_rec, 0, 100)
    mat_jacc = np.clip(mat_jacc, 0, 100)
    # accuracy 本身是 count/total，不会超 100，但加上也无妨
    list_acc = np.clip(list_acc, 0, 100)

    # === 统计聚合 (Main.m Logic) ===

    # 1. Phase Mean (Video-level -> Phase-level)
    phase_mean_jacc = np.nanmean(mat_jacc, axis=0)
    phase_std_jacc  = np.nanstd(mat_jacc, axis=0)

    phase_mean_prec = np.nanmean(mat_prec, axis=0)
    phase_std_prec  = np.nanstd(mat_prec, axis=0)

    phase_mean_rec  = np.nanmean(mat_rec, axis=0)
    phase_std_rec   = np.nanstd(mat_rec, axis=0)

    # 2. Global Mean (Phase-level -> Global)
    final_mean_jacc = np.mean(phase_mean_jacc)
    final_std_jacc  = np.std(phase_mean_jacc)

    final_mean_prec = np.mean(phase_mean_prec)
    final_std_prec  = np.std(phase_mean_prec)

    final_mean_rec  = np.mean(phase_mean_rec)
    final_std_rec   = np.std(phase_mean_rec)

    final_mean_acc = np.mean(list_acc)
    final_std_acc  = np.std(list_acc)

    # ================= 输出 =================
    print("\n" + "="*60)
    print("      MICCAI STRICT Evaluation (Python Version)")
    print("="*60)

    print(f"{'Phase':<25} | {'Jaccard':<15} | {'Precision':<15} | {'Recall':<15}")
    print("-" * 80)
    for i in range(num_phases):
        print(f"{PHASE_MAP[i]['name']:<25} | "
              f"{phase_mean_jacc[i]:.2f} ± {phase_std_jacc[i]:.2f}   | "
              f"{phase_mean_prec[i]:.2f} ± {phase_std_prec[i]:.2f}   | "
              f"{phase_mean_rec[i]:.2f} ± {phase_std_rec[i]:.2f}")

    print("-" * 80)
    print(f"Mean Accuracy:  {final_mean_acc:.2f} ± {final_std_acc:.2f}")
    print(f"Mean Jaccard:   {final_mean_jacc:.2f} ± {final_std_jacc:.2f}")
    print(f"Mean Precision: {final_mean_prec:.2f} ± {final_std_prec:.2f}")
    print(f"Mean Recall:    {final_mean_rec:.2f} ± {final_std_rec:.2f}")
    print("="*60)
    print(f"Visuals saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()