import argparse
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os

from openpi.training import config as _config
from openpi.policies import policy_config
import openpi.training.data_loader as _data

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
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载配置
    config = _config.get_config(args.config)
    # 设置评估批大小
    eval_batch_size = args.batch_size
    
    # 检查norm_stats文件是否存在
    checkpoint_path = Path(args.checkpoint_dir)
    norm_stats_path = checkpoint_path / "assets" / config.data.repo_id / "norm_stats.json"
    
    if not norm_stats_path.exists():
        print(f"\n⚠ Warning: norm_stats.json not found at {norm_stats_path}")

    # 使用官方方法创建训练好的policy - 自动检测PyTorch格式
    print("Creating trained policy...")
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    print("✓ Policy loaded and moved to device")

    # 构建数据加载器 - 参考训练代码
    print("Building data loader...")
    loader, data_config = build_datasets(config)

    # 收集预测和真实值
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():  # 评估时不需要梯度
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            # batch格式: (observation, actions)
            observation, actions = batch
            
            # 将数据移到GPU
            obs_dict = observation.to_dict()
            
            # 构建policy.infer期望的输入格式
            # Observation.from_dict() 期望的是扁平化的 "observation/xxx" 格式
            sample_batch = {}
            
            # 处理图像数据 - 关键修改：构建正确的嵌套结构
            if "image" in obs_dict:
                image_data = obs_dict["image"]
                if isinstance(image_data, dict):
                    # 将嵌套的图像字典扁平化为 observation/xxx 格式
                    for cam_key, cam_value in image_data.items():
                        sample_batch[f"observation/{cam_key}"] = cam_value.cpu().numpy()
                    
                    # 同时添加 observation/image 键（使用主相机或第一个相机）
                    # 如果有 'base_0_rgb'，使用它；否则使用第一个
                    if "base_0_rgb" in image_data:
                        sample_batch["observation/image"] = image_data["base_0_rgb"].cpu().numpy()
                    else:
                        first_cam = list(image_data.values())[0]
                        sample_batch["observation/image"] = first_cam.cpu().numpy()
                else:
                    sample_batch["observation/image"] = image_data.cpu().numpy()
            
            # 处理状态数据
            if "state" in obs_dict:
                state_data = obs_dict["state"]
                if isinstance(state_data, dict):
                    for key, value in state_data.items():
                        sample_batch[f"observation/{key}"] = value.cpu().numpy()
                else:
                    sample_batch["observation/state"] = state_data.cpu().numpy()
            
            # 处理其他可能需要的字段
            for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", 
                       "token_loss_mask", "image_mask"]:
                if key in obs_dict:
                    value = obs_dict[key]
                    if isinstance(value, torch.Tensor):
                        sample_batch[f"observation/{key}"] = value.cpu().numpy()
                    else:
                        sample_batch[f"observation/{key}"] = value
            
            # 如果有prompt，也需要处理
            if hasattr(observation, 'prompt') and observation.prompt is not None:
                sample_batch["prompt"] = observation.prompt
            elif "prompt" in obs_dict:
                sample_batch["prompt"] = obs_dict["prompt"]
            
            # 将actions转换为numpy（policy.infer返回numpy，保持一致）
            actions = actions.cpu().numpy()
            
            # 使用policy进行推理
            import pdb; pdb.set_trace()

            try:
                output = policy.infer(sample_batch)
                predictions = output["actions"]  # shape: [batch_size, action_dim]
                
                # 保留在GPU上进行计算
                all_predictions.append(predictions)
                all_ground_truth.append(actions)
                
            except Exception as e:
                print(f"\n⚠ Error processing batch {batch_idx}: {e}")
                continue
            
            # 定期清理GPU缓存
            if (batch_idx + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # 合并所有批次的结果 - 在GPU上
    predictions = torch.cat(all_predictions, dim=0)
    ground_truth = torch.cat(all_ground_truth, dim=0)
    
    print(f"\n✓ Evaluated {len(predictions)} samples")
    
    # 在GPU上计算MSE和MAE
    mse = torch.mean((predictions - ground_truth) ** 2).item()
    mae = torch.mean(torch.abs(predictions - ground_truth)).item()
    
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
        f.write(f"Batch size: {eval_batch_size}\n")
        f.write(f"Samples: {predictions.shape[0]}\n\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
    
    print(f"\n✓ Results saved to {output_dir / 'results.txt'}")
    
if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args)