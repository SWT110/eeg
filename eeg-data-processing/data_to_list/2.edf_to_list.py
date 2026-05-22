import os
import mne
import pandas as pd
from datetime import timedelta

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 设置要读取的文件夹路径和输出文件的路径
input_folder = os.path.join(script_dir, "data")
output_folder = os.path.join(script_dir, "list")

print(f"脚本目录: {script_dir}")
print(f"输入目录: {input_folder}")
print(f"输出目录: {output_folder}")

os.makedirs(output_folder, exist_ok=True)

# 获取已处理的文件列表(list目录中的xlsx文件名,不含扩展名)
processed_files = set()
if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        if file.endswith('.xlsx'):
            # 去掉.xlsx扩展名,得到原始文件名
            processed_files.add(os.path.splitext(file)[0])

print(f"\n已处理的文件数量: {len(processed_files)}")

# 统计处理情况
total_files = 0
skipped_files = 0
processed_count = 0
error_count = 0

# 遍历文件夹中的所有 EDF 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.edf'):
        total_files += 1
        base_name = os.path.splitext(filename)[0]
        
        # 检查是否已经处理过
        if base_name in processed_files:
            print(f"\n⊗ 跳过(已处理): {filename}")
            skipped_files += 1
            continue
        
        edf_path = os.path.join(input_folder, filename)

        # 读取 EDF 文件
        try:
            print(f"\n正在处理: {filename}")
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            # 获取信号数据和标签
            signal_data = raw.get_data()
            labels = raw.ch_names

            # 获取采样率和开始时间
            sfreq = raw.info['sfreq']  # 采样率(Hz)
            start_time = raw.info['meas_date']  # 获取测量开始的绝对时间
            n_samples = signal_data.shape[1]  # 样本数量
            
            # 使用采样率计算精确的时间戳(毫秒级)
            time_step = 1.0 / sfreq  # 每个样本的时间间隔(秒)
            times = [i * time_step for i in range(n_samples)]  # 精确的相对时间
            
            # 计算绝对时间(保留毫秒精度)
            if start_time is not None:
                # 移除时区信息,使其与Excel兼容
                start_time = start_time.replace(tzinfo=None)
                absolute_times = [start_time + timedelta(seconds=t) for t in times]
                # 将时间格式化为字符串,保留毫秒
                time_strings = [t.strftime('%H:%M:%S.%f')[:-3] for t in absolute_times]
                print(f"  开始时间: {start_time}")
                print(f"  采样率: {sfreq} Hz (时间精度: {time_step*1000:.3f} ms)")
                print(f"  时间格式示例: {time_strings[0]}")
            else:
                print(f"  警告: 未找到开始时间,使用相对时间")
                time_strings = [f"{t:.3f}" for t in times]

            # 创建 DataFrame
            df = pd.DataFrame(signal_data.T, columns=labels)  # 转置数据
            df['Time'] = time_strings  # 使用格式化后的时间字符串

            # 将 DataFrame 写入 Excel 文件
            output_file = os.path.join(output_folder, f'{base_name}.xlsx')
            df.to_excel(output_file, index=False)  # 不写入索引
            print(f"  ✓ 成功导出: {os.path.basename(output_file)}")
            processed_count += 1

        except Exception as e:
            print(f'  ✗ 处理 {filename} 时出错: {e}')
            error_count += 1

print("\n" + "="*50)
print("处理完成!")
print(f"总文件数: {total_files}")
print(f"已跳过: {skipped_files}")
print(f"新处理: {processed_count}")
print(f"错误数: {error_count}")
print("="*50)



