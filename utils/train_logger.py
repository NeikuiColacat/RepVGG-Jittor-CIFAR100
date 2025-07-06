import pandas as pd
import os
import time
import csv
from datetime import datetime
import threading
from collections import deque

import psutil
import GPUtil

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
            'cpu_percent': [] ,
            'gpu_percent': [],
            'gpu_mem': [],
            'cpu_mem_percent': [],
        }
        
        self.monitoring = False
        self.monitor_thread = None

    def init_csv(self):
        headers = [
            'epoch',
            'train_loss',
            'train_acc',  
            'val_loss', 
            'top1_acc',
            'top5_acc',
            'epoch_time_seconds',
            'epoch_time_formatted',
            'avg_gpu_percent',  
            'avg_cpu_percent',
            'avg_gpu_memory_used_gb',
            'avg_cpu_memory_percent',
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
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.epoch_metrics['cpu_percent'].append(cpu_percent)
                
                mem = psutil.virtual_memory()
                self.epoch_metrics['cpu_mem_percent'].append(mem.percent)
                
                gpu = GPUtil.getGPUs()[0]
                self.epoch_metrics['gpu_percent'].append(gpu.load * 100)
                self.epoch_metrics['gpu_mem'].append(gpu.memoryUsed / 1024)

                time.sleep(1) 
                    
            except Exception as e:
                print(f"线程错误: {e}")
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

        gpu_mem_list = self.epoch_metrics['gpu_mem']
        cpu_mem_list = self.epoch_metrics['cpu_mem_percent']
        gpu_percent_list = self.epoch_metrics['gpu_percent']
        cpu_percent_list = self.epoch_metrics['cpu_percent']

        avg = lambda a_list : sum(a_list) / max(1,len(a_list))

        return {
            'avg_gpu_mem_used_gb': avg(gpu_mem_list),
            'avg_gpu_percent': avg(gpu_percent_list), 
            'avg_cpu_percent': avg(cpu_percent_list),
            'avg_cpu_mem_percent': avg(cpu_mem_list),
        } 
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
    
    def log_epoch(self, epoch, train_loss, train_acc ,val_loss, top1_acc, top5_acc, 
                  epoch_time, learning_rate):
        
        self.stop_epoch_monitoring()
        
        
        metrics = self.get_epoch_avg_metrics()
        epoch_time_formatted = self.format_time(epoch_time)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row_data = [
            epoch,
            round(train_loss, 4),
            round(train_acc, 2),  
            round(val_loss, 4),
            round(top1_acc, 2),
            round(top5_acc, 2),
            round(epoch_time, 2),
            epoch_time_formatted,
            round(metrics['avg_gpu_percent'],2),  
            round(metrics['avg_cpu_percent'],2),  
            round(metrics['avg_gpu_mem_used_gb'],2),  
            round(metrics['avg_cpu_mem_percent'],2),  
            f"{learning_rate:.6f}",
            timestamp,
            self.framework
        ] 
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)