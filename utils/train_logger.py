import pandas as pd
import os
import time
import psutil
import torch
import csv
from datetime import datetime
import threading
from collections import deque

class Logger:
    def __init__(self, log_dir, framework: str, model_name: str):
        self.log_dir = log_dir
        self.framework = framework
        self.model_name = model_name
        
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, f'{framework}_{model_name}_training_log.csv')
        self.init_csv()
        
        self.process = psutil.Process(os.getpid())
        
        self.epoch_metrics = {
            'cpu_percent': deque(),
            'gpu_memory_used': deque(),
            'gpu_utilization_percent': deque(),  # 添加这一行
            'cpu_memory_used': deque(),
            'cpu_memory_percent': deque()
        }
        
        self.monitoring = False
        self.monitor_thread = None

    def init_csv(self):
        headers = [
            'epoch',
            'train_loss',
            'train_acc',  # 新增 train_acc 列
            'val_loss', 
            'top1_acc',
            'top5_acc',
            'epoch_time_seconds',
            'epoch_time_formatted',
            'avg_gpu_memory_used_gb',
            'max_gpu_memory_used_gb',
            'gpu_memory_total_gb',
            'avg_gpu_utilization_percent',  # 新增 GPU 平均占用率列
            'avg_cpu_percent',
            'max_cpu_percent',
            'avg_cpu_memory_used_gb',
            'max_cpu_memory_used_gb',
            'avg_cpu_memory_percent',
            'max_cpu_memory_percent',
            'learning_rate',
            'timestamp',
            'framework'
        ]
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)    
    
    def _monitor_system_metrics(self):
        while self.monitoring:
            try:
                # CPU 使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.epoch_metrics['cpu_percent'].append(cpu_percent)
                
                # CPU 内存使用情况
                memory_info = psutil.virtual_memory()
                cpu_memory_used_gb = memory_info.used / (1024**3)
                cpu_memory_percent = memory_info.percent
                self.epoch_metrics['cpu_memory_used'].append(cpu_memory_used_gb)
                self.epoch_metrics['cpu_memory_percent'].append(cpu_memory_percent)
                
                # GPU 内存使用情况
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                    self.epoch_metrics['gpu_memory_used'].append(gpu_memory_used)
                    
                    # GPU 平均占用率
                    gpu_utilization = torch.cuda.utilization(0)  # 获取 GPU 利用率
                    self.epoch_metrics['gpu_utilization_percent'].append(gpu_utilization)
                
                time.sleep(10)  # 每10秒采样一次
                    
            except Exception as e:
                print(f"监控线程错误: {e}")
                break
                
    
    def start_epoch_monitoring(self):
        for key in self.epoch_metrics:
            self.epoch_metrics[key].clear()
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_epoch_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def get_epoch_avg_metrics(self):
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if self.epoch_metrics['gpu_memory_used']:
                avg_gpu_memory = sum(self.epoch_metrics['gpu_memory_used']) / len(self.epoch_metrics['gpu_memory_used'])
                max_gpu_memory = max(self.epoch_metrics['gpu_memory_used'])
            else:
                avg_gpu_memory = max_gpu_memory = 0
            
            # GPU 平均占用率
            if self.epoch_metrics['gpu_utilization_percent']:
                avg_gpu_utilization = sum(self.epoch_metrics['gpu_utilization_percent']) / len(self.epoch_metrics['gpu_utilization_percent'])
            else:
                avg_gpu_utilization = 0
        else:
            gpu_memory_total = avg_gpu_memory = max_gpu_memory = avg_gpu_utilization = 0
        
        # CPU 指标
        if self.epoch_metrics['cpu_percent']:
            avg_cpu_percent = sum(self.epoch_metrics['cpu_percent']) / len(self.epoch_metrics['cpu_percent'])
            max_cpu_percent = max(self.epoch_metrics['cpu_percent'])
        else:
            avg_cpu_percent = max_cpu_percent = 0
        
        if self.epoch_metrics['cpu_memory_used']:
            avg_cpu_memory_used = sum(self.epoch_metrics['cpu_memory_used']) / len(self.epoch_metrics['cpu_memory_used'])
            max_cpu_memory_used = max(self.epoch_metrics['cpu_memory_used'])
        else:
            avg_cpu_memory_used = max_cpu_memory_used = 0
        
        if self.epoch_metrics['cpu_memory_percent']:
            avg_cpu_memory_percent = sum(self.epoch_metrics['cpu_memory_percent']) / len(self.epoch_metrics['cpu_memory_percent'])
            max_cpu_memory_percent = max(self.epoch_metrics['cpu_memory_percent'])
        else:
            avg_cpu_memory_percent = max_cpu_memory_percent = 0
        
        return {
            'avg_gpu_memory_used_gb': avg_gpu_memory,
            'max_gpu_memory_used_gb': max_gpu_memory,
            'gpu_memory_total_gb': gpu_memory_total,
            'avg_gpu_utilization_percent': avg_gpu_utilization,  # 返回 GPU 平均占用率
            'avg_cpu_percent': avg_cpu_percent,
            'max_cpu_percent': max_cpu_percent,
            'avg_cpu_memory_used_gb': avg_cpu_memory_used,
            'max_cpu_memory_used_gb': max_cpu_memory_used,
            'avg_cpu_memory_percent': avg_cpu_memory_percent,
            'max_cpu_memory_percent': max_cpu_memory_percent
        } 
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
    
    def log_epoch(self, epoch, train_loss, train_acc ,val_loss, top1_acc, top5_acc, 
                  epoch_time, learning_rate):
        
        self.stop_epoch_monitoring()
        
        
        system_metrics = self.get_epoch_avg_metrics()
        epoch_time_formatted = self.format_time(epoch_time)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row_data = [
            epoch,
            round(train_loss, 4),
            round(train_acc, 2),  # 新增 train_acc
            round(val_loss, 4),
            round(top1_acc, 2),
            round(top5_acc, 2),
            round(epoch_time, 2),
            epoch_time_formatted,
            round(system_metrics['avg_gpu_memory_used_gb'], 2),
            round(system_metrics['max_gpu_memory_used_gb'], 2),
            round(system_metrics['gpu_memory_total_gb'], 2),
            round(system_metrics['avg_gpu_utilization_percent'], 1),  # 新增 GPU 平均占用率
            round(system_metrics['avg_cpu_percent'], 1),
            round(system_metrics['max_cpu_percent'], 1),
            round(system_metrics['avg_cpu_memory_used_gb'], 2),
            round(system_metrics['max_cpu_memory_used_gb'], 2),
            round(system_metrics['avg_cpu_memory_percent'], 1),
            round(system_metrics['max_cpu_memory_percent'], 1),
            f"{learning_rate:.6f}",
            timestamp,
            self.framework
        ] 
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
    def load_log_as_dataframe(self):
        if os.path.exists(self.csv_file):
            return pd.read_csv(self.csv_file)
        else:
            return pd.DataFrame()
    
    def get_summary_stats(self):
        df = self.load_log_as_dataframe()
        if df.empty:
            return None
        
        summary = {
            'total_epochs': len(df),
            'best_top1_acc': df['top1_acc'].max(),
            'best_top5_acc': df['top5_acc'].max(),
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_val_loss': df['val_loss'].iloc[-1],
            'avg_epoch_time': df['epoch_time_seconds'].mean(),
            'avg_gpu_memory': df['avg_gpu_memory_used_gb'].mean(),
            'avg_cpu_percent': df['avg_cpu_percent'].mean(),
            'avg_cpu_memory_percent': df['avg_cpu_memory_percent'].mean(),
            'framework': self.framework
        }
        
        return summary
    
    def print_summary(self):
        summary = self.get_summary_stats()
        if summary:
            print(f"\n{'='*60}")
            print(f"Training Summary - {self.framework}")
            print(f"{'='*60}")
            print(f"Total Epochs: {summary['total_epochs']}")
            print(f"Best Top-1 Acc: {summary['best_top1_acc']:.2f}%")
            print(f"Best Top-5 Acc: {summary['best_top5_acc']:.2f}%")
            print(f"Final Train Loss: {summary['final_train_loss']:.4f}")
            print(f"Final Val Loss: {summary['final_val_loss']:.4f}")
            print(f"Avg Epoch Time: {self.format_time(summary['avg_epoch_time'])}")
            print(f"Avg GPU Memory: {summary['avg_gpu_memory']:.2f}GB")
            print(f"Avg CPU Usage: {summary['avg_cpu_percent']:.1f}%")
            print(f"Avg RAM Usage: {summary['avg_cpu_memory_percent']:.1f}%")
            print(f"{'='*60}")