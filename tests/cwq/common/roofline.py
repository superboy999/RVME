'''
Author: superboy
Date: 2024-07-10 23:09:24
LastEditTime: 2024-07-11 15:01:10
LastEditors: superboy
Description: 
FilePath: /gem5-rvm/tests/cwq/common/roofline.py

'''
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
# 定义计算密度（FLOP/byte）
operational_intensity = np.logspace(-2, 3, 100)

# 定义计算性能和内存带宽的峰值
peak_performance = 1024  # GFLOPS
peak_bandwidth = 32  # GB/s

# 计算Roofline
compute_bound = np.full_like(operational_intensity, peak_performance)
memory_bound = peak_bandwidth * operational_intensity

# 找到交汇点
intersection_index = np.where(memory_bound >= peak_performance)[0][0]
intersection_intensity = operational_intensity[intersection_index]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制计算性能和内存带宽的限制线（部分）
plt.plot(operational_intensity[:intersection_index], memory_bound[:intersection_index], label='Peak Bandwidth', color='b')
plt.plot(operational_intensity[intersection_index:], compute_bound[intersection_index:], label='Peak Performance', color='r')

# 填充Roofline区域
plt.fill_between(operational_intensity[:intersection_index], 0, memory_bound[:intersection_index], alpha=0.1, color='blue')
plt.fill_between(operational_intensity[intersection_index:], 0, compute_bound[intersection_index:], alpha=0.1, color='red')

# 添加点（示例点）
example_intensity = [4, 8, 16, 32, 64, 128, 256, 512]
example_performance = [48.76, 97.5, 146.28, 195, 243, 292.5, 341.33, 390.09]
plt.scatter(example_intensity, example_performance, color='k', zorder=5)
for i in range(len(example_intensity)):
    plt.text(example_intensity[i], example_performance[i]*1.1, f'Point {i+1}', ha='center')

# 设置图表属性
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (OPs/byte)')
plt.ylabel('Performance (GFLOPS)')
plt.title('Roofline Model')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axvline(x=1, color='gray', linestyle='--', linewidth=0.5)

# 显示图表
plt.show()
