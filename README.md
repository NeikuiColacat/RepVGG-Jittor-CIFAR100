# RepVGG-MAOMAO-implementation


指标	PyTorch	Jittor
Top-1 Accuracy
Top-5 Accuracy	
最终 Loss	1.21	1.23
每个 epoch 时间	3m 12s	3m 28s
每个epoch 显存占用率	11.2 GB	11.4 GB
每个epoch cpu 占用率 
每个epoch cpu内存占用率
每个epoch train loss
每个epoch val loss
每个epoch top-1 acc , top-5 acc



混合精度 AMP	autocast() + GradScaler	×1.5–2×
PyTorch 编译	model = torch.compile(model)	×1.1–1.4×
内存布局	channels_last + TF32	×1.2×
DataLoader	增 num_workers, prefetch_factor, 本地 NVMe	×1.1×
CPU 线程调度	OMP_NUM_THREADS=15, torch.set_num_threads(…)	避免掉帧
梯度累积	模拟更大 batch	提高泛化
