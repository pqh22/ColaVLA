#!/usr/bin/env python
"""
将vla_results目录下的所有预测结果整合成一个pkl文件
格式: {sample_token: trajectory_array(6,2)}
"""

import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def merge_vla_results(vla_results_dir, output_path):
    """
    合并vla_results目录下的所有pkl文件
    
    Args:
        vla_results_dir: vla_results目录路径
        output_path: 输出pkl文件路径
    """
    vla_results_dir = Path(vla_results_dir)
    
    # 检查目录是否存在
    if not vla_results_dir.exists():
        raise ValueError(f"Directory does not exist: {vla_results_dir}")
    
    # 获取所有pkl文件
    pkl_files = list(vla_results_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pkl files in {vla_results_dir}")
    
    if len(pkl_files) == 0:
        raise ValueError(f"No pkl files found in {vla_results_dir}")
    
    # 创建结果字典
    merged_results = {}
    
    # 遍历所有pkl文件
    for pkl_file in tqdm(pkl_files, desc="Merging predictions"):
        # 获取sample_token（文件名去掉.pkl后缀）
        sample_token = pkl_file.stem
        
        try:
            # 读取预测轨迹
            with open(pkl_file, 'rb') as f:
                trajectory = pickle.load(f)
            
            # 确保是numpy数组
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            
            # 检查形状
            if trajectory.shape != (6, 2):
                print(f"Warning: {sample_token} has shape {trajectory.shape}, expected (6, 2)")
                # 尝试reshape或者跳过
                if trajectory.size == 12:
                    trajectory = trajectory.reshape(6, 2)
                else:
                    print(f"Skipping {sample_token} due to incompatible shape")
                    continue
            
            # 添加到结果字典
            merged_results[sample_token] = trajectory
            
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            continue
    
    print(f"\nSuccessfully merged {len(merged_results)} predictions")
    
    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(merged_results, f)
    
    print(f"Saved merged results to: {output_path}")
    
    # 验证保存的文件
    with open(output_path, 'rb') as f:
        loaded_results = pickle.load(f)
    
    print(f"\nVerification:")
    print(f"  Total samples: {len(loaded_results)}")
    
    # 检查一个样本
    if len(loaded_results) > 0:
        sample_token = list(loaded_results.keys())[0]
        sample_traj = loaded_results[sample_token]
        print(f"  Sample token: {sample_token}")
        print(f"  Sample trajectory shape: {sample_traj.shape}")
        print(f"  Sample trajectory dtype: {sample_traj.dtype}")
    
    return merged_results


def main():
    parser = argparse.ArgumentParser(description='Merge VLA prediction results into a single pkl file')
    parser.add_argument(
        '--vla-results-dir',
        type=str,
        default='/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/results_planning/aa_epoch10_msarplv2_ms6_seqformer_wckpt_fullcontext_regw80_pretraintraj_globalreason_correctidx_top3pred/vla_results',
        help='Path to vla_results directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/nfs/dataset-ofs-voyager-research/pqh/OmniDrive/evaluation/my_vlm_results.pkl',
        help='Output pkl file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Merging VLA Prediction Results")
    print("=" * 80)
    print(f"Input directory: {args.vla_results_dir}")
    print(f"Output file: {args.output}")
    print("=" * 80)
    
    merged_results = merge_vla_results(args.vla_results_dir, args.output)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()

