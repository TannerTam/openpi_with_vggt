import argparse
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.policies import policy_config_with_vggt
import openpi.training.data_loader as _data

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OpenPI checkpoint accuracy')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Config name'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eval_results',
        help='Directory to save results (default: ./eval_results)'
    )
    return parser.parse_args()


def plot_comparison(prediction, ground_truth, output_dir):
    """绘制单个样本的预测vs真实值对比图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 处理action chunk (seq_len, action_dim) 或 single action (action_dim,)
    if len(prediction.shape) > 1:
        # Action chunk: 显示所有时间步
        seq_len, action_dim = prediction.shape
        x = np.arange(seq_len)
        
        for dim in range(action_dim):
            ax.plot(x, ground_truth[:, dim], 'o-', label=f'GT Dim {dim}', linewidth=2, markersize=6)
            ax.plot(x, prediction[:, dim], 'x--', label=f'Pred Dim {dim}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Step', fontsize=12)
    else:
        # Single action: 条形图对比
        action_dim = len(prediction)
        x = np.arange(action_dim)
        width = 0.35
        
        ax.bar(x - width/2, ground_truth, width, label='Ground Truth', alpha=0.8)
        ax.bar(x + width/2, prediction, width, label='Prediction', alpha=0.8)
        ax.set_xlabel('Action Dimension', fontsize=12)
        ax.set_xticks(x)
    
    mae = np.mean(np.abs(prediction - ground_truth))
    mse = np.mean((prediction - ground_truth) ** 2)
    
    ax.set_ylabel('Action Value', fontsize=12)
    ax.set_title(f'Prediction vs Ground Truth\nMAE: {mae:.4f}, MSE: {mse:.6f}', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'prediction_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def evaluate_checkpoint(args):
    """主评估函数"""
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    
    # 加载配置
    config = _config.get_config(args.config)
    action_horizon = config.model.action_horizon
    
    # 检查norm_stats文件是否存在
    checkpoint_path = Path(args.checkpoint_dir)
    norm_stats_path = checkpoint_path / "assets" / config.data.repo_id / "norm_stats.json"
    
    if not norm_stats_path.exists():
        print(f"\n⚠ Warning: norm_stats.json not found at {norm_stats_path}")

    # 使用官方方法创建训练好的policy
    print("Creating trained policy...")
    policy = policy_config_with_vggt.create_trained_policy(config, args.checkpoint_dir)
    print("✓ Policy loaded and moved to device")

    # 构建数据加载器
    print("Building data loader...")
    dataset = LeRobotDataset(
        repo_id="lerobot/franka_cam10",
        root="/home/tanner/Downloads/lerobot_format/franka_cam10",
    )
    print(f"✓ Dataset loaded with {len(dataset)} samples")

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=eval_batch_size,
    #     shuffle=False,
    #     num_workers=4,  # 并行加载数据
    #     pin_memory=True  # 加速GPU传输
    # )

    # example = dataset[0]

    # formatted_example = {
    #     "observation/image": example["observation.images.camera"],
    #     "observation/state": example["observation.state"],
    #     "actions": example["actions"],
    #     "prompt": example.get("language_instruction", "")
    # }

    # import pdb; pdb.set_trace()

    # result = policy.infer(formatted_example)
    # predicted_actions = result["actions"]

    # 收集预测和真实值
    all_predictions = []
    all_ground_truth = []
    action_dim = 7

    # 遍历数据集（批量并行）
    for idx in range(len(dataset) - action_horizon):
    # for idx in range(10):
        example = dataset[idx]

        formatted_example = {
            "observation/image": example["observation.images.camera"],
            "observation/state": example["observation.state"],
            # "actions": example["actions"],
            "prompt": example.get("language_instruction", "")
        }

        # ===== 推理 =====
        result = policy.infer(formatted_example)
        predicted_actions = result["actions"]

        # ✅ 统一转成 torch.Tensor
        if not isinstance(predicted_actions, torch.Tensor):
            predicted_actions = torch.as_tensor(predicted_actions)

        # 确保形状是 (16, 7)
        assert predicted_actions.shape == (action_horizon, action_dim), \
            f"predicted_actions shape error: {predicted_actions.shape}"

        # ===== 构造 GT =====
        gt_actions = []

        for t in range(1, action_horizon + 1):
            a = dataset[idx + t]["actions"]

            # 转 tensor
            if not isinstance(a, torch.Tensor):
                a = torch.as_tensor(a)

            # 只取最后 7 维
            a = a[-action_dim:]
            gt_actions.append(a)

        # (16, 7)
        gt_actions = torch.stack(gt_actions, dim=0)

        # ===== append =====
        all_predictions.append(predicted_actions)
        all_ground_truth.append(gt_actions)

        # ===== 打印进度 =====
        if (idx + 1) % 10 == 0:
            print(f"Processed {(idx + 1)} / {len(dataset) - action_horizon} samples...")

    # ===== 合并 =====
    predictions = torch.cat(all_predictions, dim=0)      # (N*16, 7)
    ground_truth = torch.cat(all_ground_truth, dim=0)    # (N*16, 7)

    print(f"\n✓ Evaluated {predictions.shape[0]} actions")

    # ===== 计算指标 =====
    mse = torch.mean((predictions - ground_truth) ** 2).item()
    mae = torch.mean(torch.abs(predictions - ground_truth)).item()

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples: {len(predictions)}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print("=" * 50)
    
    # 转换到CPU用于绘图
    predictions_cpu = predictions.cpu().numpy()
    ground_truth_cpu = ground_truth.cpu().numpy()
    
    # 随机挑选一个样本绘图
    if len(predictions_cpu) > 0:
        random_idx = np.random.randint(0, len(predictions_cpu))
        print(f"\nPlotting sample #{random_idx}...")
        plot_comparison(predictions_cpu[random_idx], ground_truth_cpu[random_idx], args.output_dir)
    
    # 保存结果
    checkpoint_path = Path(args.checkpoint_dir)
    parts = checkpoint_path.parts[-3:]
    output_subdir = Path(*parts) if len(parts) == 3 else checkpoint_path.name
    output_dir = Path(args.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint_dir}\n")
        # f.write(f"Batch size: {eval_batch_size}\n")
        f.write(f"Samples: {predictions.shape[0]}\n\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
    
    print(f"\n✓ Results saved to {output_dir / 'results.txt'}")
    
if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args)