import sys
import os#同步

# 修改1：获取当前脚本所在的绝对路径，并将 core 目录添加到 sys.path
# 这样无论在哪个目录下运行脚本，都能正确找到 core 中的模块，同时解决服务器路径问题
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'core'))

import argparse
import cv2
import numpy as np
import torch
import re
import time
from tqdm import tqdm
from PIL import Image

# 修改2：去除 core. 前缀，直接导入，以配合 RAFT 源码内部的 import 逻辑
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


class Config:
    # ==================== 动态路径配置（适配服务器） ====================
    # 获取当前脚本所在目录 (即 RAFT-master)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 推导上一级目录 (即 code_80)
    CODE_ROOT = os.path.dirname(CURRENT_DIR)
    # 数据集根目录 (假设 data 在 code_80 下)
    DATA_ROOT = os.path.join(CODE_ROOT, 'data', 'cholec80')

    CHOLEC80_ROOT = os.path.join(DATA_ROOT, 'cutMargin')         # 原始帧目录
    FLOW_VIS_SAVE = os.path.join(DATA_ROOT, 'raft_flow_vis')     # 可视化保存目录
    FLOW_NPY_SAVE = os.path.join(DATA_ROOT, 'raft_flow_npy')     # npy保存目录
    CKPT_PATH = os.path.join(CURRENT_DIR, 'models', 'raft-things.pth') # 模型权重
    LOG_PATH = os.path.join(DATA_ROOT, 'raft_flow_process.log')

    # ==================== 核心参数 ====================
    FRAME_INTERVAL = 25  # 按帧编号间隔25帧（0→25、25→50...）

    # ==================== 后处理参数 ====================
    FLOW_MEDIAN_FILTER = False  # 是否对光流进行中值滤波
    MEDIAN_KERNEL = 5

    # ==================== 设备 ====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_RAW_FLOW = True  # 强制保存原始光流（核心需求）
    SAVE_VIS_IMG = False  # 修改：不再生成可视化图片以节省空间


def extract_frame_num(filename):
    """精准提取帧文件名中的数字编号（适配Cholec80命名：0.jpg/25.jpg/frame_0050.jpg）"""
    basename = os.path.splitext(filename)[0]
    nums = re.findall(r'\d+', basename)
    return int(nums[-1]) if nums else -1


def median_filter_flow(flow_np, kernel_size=5):
    """对光流的x,y分量分别进行中值滤波"""
    flow_x = cv2.medianBlur(flow_np[:, :, 0], kernel_size)
    flow_y = cv2.medianBlur(flow_np[:, :, 1], kernel_size)
    return np.stack([flow_x, flow_y], axis=2)


def load_image(imfile, device):
    """加载单张图像并转为张量（与RAFT demo一致）"""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0).to(device)


def compute_flow(model, img1_path, img2_path, device, config):
    """给定两帧路径，计算原始光流并返回（含可选中值滤波）"""
    with torch.no_grad():  # 禁用梯度计算，节省显存
        # 加载图像
        image1 = load_image(img1_path, device)
        image2 = load_image(img2_path, device)

        # RAFT需要pad到8的倍数
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)

        # 推理光流
        _, flow_up = model(image1_pad, image2_pad, iters=20, test_mode=True)  # flow_up: [1,2,H,W]

        # 去除padding，恢复原始尺寸
        flow = padder.unpad(flow_up)

        # 转换为numpy数组 (H,W,2)
        flow_np = flow[0].cpu().permute(1, 2, 0).numpy()

        # 可选：中值滤波
        if config.FLOW_MEDIAN_FILTER:
            flow_np = median_filter_flow(flow_np, kernel_size=config.MEDIAN_KERNEL)

        # 生成可视化图（BGR格式，适配cv2保存）
        flow_bgr = None
        if config.SAVE_VIS_IMG:
            flow_rgb = flow_viz.flow_to_image(flow_np)
            flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

        return flow_np, flow_bgr


if __name__ == "__main__":
    config = Config()
    device = config.DEVICE
    print(f"使用设备：{device}")

    # -------------------- 加载RAFT模型 --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # 手动指定模型参数
    args.model = config.CKPT_PATH
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    # 加载模型（修复多卡权重和设备适配：先DataParallel再to(device)）
    model = RAFT(args)
    state_dict = torch.load(args.model, map_location=device)
    # 处理多卡训练的权重前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()  # 先设为评估模式
    # 多卡加速（可选）
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)  # 最后统一移到设备
    print("RAFT模型加载成功！")

    # -------------------- 创建输出目录 --------------------
    if config.SAVE_VIS_IMG:
        os.makedirs(config.FLOW_VIS_SAVE, exist_ok=True)
    os.makedirs(config.FLOW_NPY_SAVE, exist_ok=True)

    total_success = 0
    total_fail = 0

    # -------------------- 初始化日志（with语句自动关闭） --------------------
    with open(config.LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== RAFT任务开始：{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_file.write(f"帧间隔设置：{config.FRAME_INTERVAL}帧\n")
        log_file.write(f"中值滤波：{'开启' if config.FLOW_MEDIAN_FILTER else '关闭'}\n")
        log_file.write(f"生成可视化图片：{'是' if config.SAVE_VIS_IMG else '否 (已关闭)'}\n")
        log_file.write(f"光流文件命名规则：源帧编号（如0→25帧对生成0.npy）\n")  # 新增日志说明
        log_file.flush()

        # 处理视频1~80
        for video_id in range(1, 81):
            video_frame_path = os.path.join(config.CHOLEC80_ROOT, str(video_id))
            video_vis_save = os.path.join(config.FLOW_VIS_SAVE, str(video_id))
            video_npy_save = os.path.join(config.FLOW_NPY_SAVE, str(video_id))
            if config.SAVE_VIS_IMG:
                os.makedirs(video_vis_save, exist_ok=True)
            os.makedirs(video_npy_save, exist_ok=True)

            video_start_time = time.time()
            success_cnt = 0
            fail_cnt = 0
            pbar = None

            # 检查视频目录是否存在
            if not os.path.exists(video_frame_path):
                warn_msg = f"警告：视频{video_id}文件夹不存在，跳过！"
                print(warn_msg)
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {warn_msg}\n")
                log_file.flush()
                fail_cnt += 1
                total_fail += fail_cnt
                continue

            # 1. 构建「帧编号→文件名」字典（精准映射）
            frame_info = {}
            all_files = os.listdir(video_frame_path)
            for f in all_files:
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    num = extract_frame_num(f)
                    if num != -1:
                        frame_info[num] = f

            # 检查是否有有效帧文件
            if len(frame_info) < 2:
                warn_msg = f"警告：视频{video_id}有效帧文件不足2个（共{len(all_files)}个文件），跳过！"
                print(warn_msg)
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {warn_msg}\n")
                log_file.flush()
                fail_cnt += len(frame_info)
                total_fail += fail_cnt
                continue

            # 2. 按帧编号升序排列
            sorted_nums = sorted(frame_info.keys())
            total_frames = len(sorted_nums)

            # 3. 筛选「编号+间隔」的有效帧对（核心修复：按编号匹配）
            valid_pairs = []
            for src_num in sorted_nums:
                tgt_num = src_num + config.FRAME_INTERVAL
                if tgt_num in frame_info:
                    valid_pairs.append((src_num, tgt_num, frame_info[src_num], frame_info[tgt_num]))

            valid_pair_num = len(valid_pairs)
            if valid_pair_num < 1:
                warn_msg = f"警告：视频{video_id}无有效帧对（总帧数{total_frames}，帧间隔{config.FRAME_INTERVAL}），跳过！"
                print(warn_msg)
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {warn_msg}\n")
                log_file.flush()
                fail_cnt += total_frames
                total_fail += fail_cnt
                continue

            print(f"\n开始处理视频{video_id}，共{total_frames}有效帧，{valid_pair_num}个有效帧对...")

            # 扫描已存在的原始光流文件（修改1：按源帧编号判断，而非目标帧）
            existing_npy = set()
            if os.path.exists(video_npy_save):
                for f in os.listdir(video_npy_save):
                    if f.endswith('.npy'):  # 简化匹配，直接提取文件名中的数字
                        src_num = extract_frame_num(f)
                        if src_num != -1:
                            existing_npy.add(src_num)
            print(f"视频{video_id}：已存在 {len(existing_npy)} 个原始光流文件，将跳过它们")
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 视频{video_id}：{total_frames}有效帧，{valid_pair_num}有效帧对，{len(existing_npy)}个已处理帧对\n")
            log_file.flush()

            # 处理有效帧对
            pbar = tqdm(valid_pairs, desc=f"视频{video_id}帧对处理")
            for src_num, tgt_num, src_file, tgt_file in pbar:
                # 跳过已处理的帧对（修改2：按源帧编号判断）
                if src_num in existing_npy:
                    pbar.set_postfix({"跳过": f"{src_num}→{tgt_num}"})
                    success_cnt += 1
                    total_success += 1
                    continue

                pbar.set_postfix({"当前帧对": f"{src_num}→{tgt_num}"})

                img1_path = os.path.join(video_frame_path, src_file)
                img2_path = os.path.join(video_frame_path, tgt_file)

                try:
                    # 计算原始光流和可视化图
                    flow_np, flow_bgr = compute_flow(model, img1_path, img2_path, device, config)

                    # 保存原始光流（修改3：按源帧编号命名，核心改动）
                    npy_save_path = os.path.join(video_npy_save, f"{src_num}.npy")
                    np.save(npy_save_path, flow_np.astype(np.float32))

                    # 保存可视化图（保持原有命名，便于关联帧对）
                    if config.SAVE_VIS_IMG and flow_bgr is not None:
                        vis_save_path = os.path.join(video_vis_save, f"{src_num}_{tgt_num}_flow.png")
                        cv2.imwrite(vis_save_path, flow_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    success_cnt += 1
                    total_success += 1

                    # 帧对级显存清理（防止显存泄漏）
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                except Exception as e:
                    err_msg = f"视频{video_id}帧对{src_num}→{tgt_num}处理失败：{str(e)}"
                    print(f"\n{err_msg}，跳过！")
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {err_msg}\n")
                    log_file.flush()
                    fail_cnt += 1
                    total_fail += 1
                    continue

            if pbar is not None:
                pbar.close()

            # 记录视频处理日志
            video_cost = time.time() - video_start_time
            video_log = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 视频{video_id}："
                         f"成功{success_cnt}帧对，失败{fail_cnt}帧对，耗时{video_cost:.2f}s\n")
            log_file.write(video_log)
            log_file.flush()
            print(f"\n视频{video_id}处理完成！"
                  f"\n- 光流可视化图：{video_vis_save if config.SAVE_VIS_IMG else '未生成'}"
                  f"\n- 原始光流npy：{video_npy_save}"
                  f"\n- 耗时：{video_cost:.2f}秒")

            # 视频级显存清理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # 最终统计
        log_file.write(f"\n===== 任务结束：{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_file.write(f"总处理结果：成功{total_success}帧对，失败{total_fail}帧对\n")
        log_file.write(f"光流可视化根路径：{config.FLOW_VIS_SAVE}\n")
        log_file.write(f"原始光流npy根路径：{config.FLOW_NPY_SAVE}\n")
        log_file.flush()

    # 控制台最终输出
    print("\n" + "=" * 60)
    print("所有视频处理完成！")
    print(f"总统计：成功{total_success}帧对，失败{total_fail}帧对")
    print(f"光流可视化根路径：{config.FLOW_VIS_SAVE}")
    print(f"原始光流npy根路径：{config.FLOW_NPY_SAVE}")
    print(f"处理日志：{config.LOG_PATH}")
    print("=" * 60)