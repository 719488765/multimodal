import subprocess
import json
import os
import sys
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd):
    """运行Shell命令并捕获输出"""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(f"Stderr: {stderr}")
            return False, stderr
        
        return True, stdout
    except Exception as e:
        logger.error(f"Exception executing command: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="自动运行基线实验")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='results/baselines', help='结果保存目录')
    parser.add_argument('--python_path', type=str, default='python', help='Python解释器路径')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义基线实验列表
    # 注意：这里假设train.py可以通过命令行参数覆盖配置，或者使用不同的配置文件
    # 实际上为了简单起见，这里演示如何生成不同配置并运行
    
    experiments = [
        {'name': 'video_only', 'modals': ['video']},
        {'name': 'audio_only', 'modals': ['audio']},
        {'name': 'text_only', 'modals': ['text']},
        {'name': 'physiological_only', 'modals': ['physiological']},
        {'name': 'video_audio', 'modals': ['video', 'audio']},
        {'name': 'all_modals', 'modals': ['video', 'audio', 'text', 'physiological']}
    ]

    summary_results = {}

    for exp in experiments:
        exp_name = exp['name']
        logger.info(f"Starting experiment: {exp_name}")
        
        # 1. 创建临时配置文件或使用命令行参数覆盖（假设train.py支持--modals参数）
        # 这里为了演示，我们假设train.py接受--active_modals参数来动态激活模态
        
        cmd = [
            args.python_path, 'scripts/train.py',
            '--config', args.config,
            '--mode', 'pretrain',
            # 假设train.py支持这个参数，或者我们需要修改train.py来支持它
            # 如果train.py不支持，正确做法是先生成对应的yaml文件
        ]
        
        # 实际操作：我们修改config.yaml或者创建临时的yaml
        # 这里模拟创建一个临时的yaml文件
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
            
        # 修改配置以仅启用特定模态
        # 注意：这取决于代码如何处理模态缺失，这里假设将其他模态路径设为None或在模型初始化时控制
        # 这是一个简化处理，实际需要train.py配合
        temp_config_path = os.path.join(args.output_dir, f'config_{exp_name}.yaml')
        
        # 标记活跃模态（这里仅作为元数据写入，实际train.py需要逻辑去读取它，或者我们在train.py中通过active_modals参数控制）
        base_config['active_modals'] = exp['modals'] 
        
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f)
            
        cmd = [
            args.python_path, 'scripts/train.py',
            '--config', temp_config_path,
            '--mode', 'pretrain'
        ]
        
        success, output = run_command(cmd)
        
        if success:
            logger.info(f"Experiment {exp_name} completed successfully.")
            # 假设train.py最后输出了结果或者保存了results.json
            # 这里简单记录 success
            summary_results[exp_name] = "Success"
        else:
            logger.error(f"Experiment {exp_name} failed.")
            summary_results[exp_name] = "Failed"

    # 保存汇总结果
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=4)
        
    logger.info("All baseline experiments finished.")

if __name__ == "__main__":
    main()
