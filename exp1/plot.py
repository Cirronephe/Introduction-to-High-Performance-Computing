import matplotlib.pyplot as plt
import numpy as np
# 数据

"""
message_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                  16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 
                  4194304]
latencies = [1.16, 1.11, 1.09, 1.08, 1.08, 1.14, 1.25, 1.33, 1.91, 1.98, 
             2.25, 2.72, 3.65, 5.19, 7.04, 9.63, 15.03, 24.82, 32.01, 56.73, 
             99.99, 186.84, 361.40]
bandwidths = [4.31, 8.44, 17.56, 34.98, 69.95, 140.90, 245.70, 478.29, 1016.90, 
              1900.09, 2926.51, 4542.62, 5929.24, 6149.52, 10325.16, 10999.94, 
              11593.71, 11842.04, 11943.14, 12010.84, 12041.30, 12054.20, 12058.70]
"""

message_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                  16384, 32768]
latencies = [44.84, 45.02, 44.95, 45.10, 45.17, 45.42, 46.38, 47.88, 52.63, 58.38, 
             70.30, 87.70, 109.73, 185.40, 214.37, 363.11]
bandwidths = [0.31, 0.61, 1.22, 2.40, 5.14, 9.28, 16.79, 33.16, 55.66, 80.94, 
              95.89, 105.26, 110.94, 114.19, 115.97, 116.76]

# 创建图形
plt.figure(figsize=(12, 5))
# 绘制延迟图
plt.subplot(1, 2, 1)
plt.plot(message_lengths, latencies, 'b-o', markersize=5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Latency vs Message Length')
plt.xlabel('Message Length (Bytes)')
plt.ylabel('Latency (μs)')
# 绘制带宽图
plt.subplot(1, 2, 2)
plt.plot(message_lengths, bandwidths, 'r-o', markersize=5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Bandwidth vs Message Length')
plt.xlabel('Message Length (Bytes)')
plt.ylabel('Bandwidth (MB/s)')
# 调整布局
plt.tight_layout()
# 保存图片
plt.savefig('latency_bandwidth_plots_linear.png', dpi=300, bbox_inches='tight')
plt.show()
