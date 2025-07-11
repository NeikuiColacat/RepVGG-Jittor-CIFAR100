import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

def plot_training_logs(csv_path, output_dir):
    try:
        # 加载CSV文件
        data = pd.read_csv(csv_path)
        
        # 获取文件名（不含扩展名）作为图表标题前缀
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        pos = filename.find('_')
        filename = filename[pos + 1 : ]
        pos = filename.rfind('_')
        filename = filename[:pos]
        pos = filename.rfind('_')
        filename = filename[:pos]
        
        # 1. 绘制Loss曲线
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='epoch', y='train_loss', data=data, label='Train Loss', color='blue', marker='o', markersize=3)
        sns.lineplot(x='epoch', y='val_loss', data=data, label='Validation Loss', color='orange', marker='s', markersize=3)
        plt.title(f'{filename} - Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存Loss图
        loss_output_path = os.path.join(output_dir, f'{filename}_loss.png')
        plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 绘制Accuracy曲线
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='epoch', y='train_acc', data=data, label='Train Accuracy', color='green', marker='o', markersize=3)
        sns.lineplot(x='epoch', y='top1_acc', data=data, label='Validation Accuracy (Top-1)', color='red', marker='s', markersize=3)
        plt.title(f'{filename} - Training and Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存Accuracy图
        acc_output_path = os.path.join(output_dir, f'{filename}_accuracy.png')
        plt.savefig(acc_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 绘制学习率曲线（如果存在）
        if 'learning_rate' in data.columns:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='epoch', y='learning_rate', data=data, color='purple', marker='o', markersize=3)
            plt.title(f'{filename} - Learning Rate Schedule', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存学习率图
            lr_output_path = os.path.join(output_dir, f'{filename}_learning_rate.png')
            plt.savefig(lr_output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"success: {csv_path}")
        print(f"save to: {output_dir}")
        
    except Exception as e:
        print(f"fail: {csv_path}")
        print(f"faile msg: {str(e)}")

def main():
    # 当前工作目录
    current_dir = "/root/autodl-tmp/logs"
    
    # 查找所有子文件夹中的CSV文件
    csv_pattern = os.path.join(current_dir, "*", "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    for csv_file in csv_files:
        print(f"   - {csv_file}")
    
    for csv_file in csv_files:
        # 获取CSV文件所在的目录
        csv_dir = os.path.dirname(csv_file)
        
        # 在同一目录下保存图片
        plot_training_logs(csv_file, csv_dir)
    
    print(f"done")

if __name__ == "__main__":
    main()