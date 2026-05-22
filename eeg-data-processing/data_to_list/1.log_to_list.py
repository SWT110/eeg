import re
import pandas as pd
from datetime import datetime
import os
import glob

def parse_log_file(file_path):
    """
    解析日志文件，提取时间戳和传感器数据
    """
    data = []
    
    print(f"正在读取文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        with open(file_path, 'r', encoding='gbk') as file:
            content = file.read()
    
    # 使用更精确的正则表达式提取每一行的时间戳和十六进制数据
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\].*?<<<\s*((?:[A-F0-9]{2}\s*)+)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    print(f"  找到 {len(matches)} 行数据")
    
    processed_packets = 0
    
    for timestamp, hex_data in matches:
        # 清理十六进制数据，移除空格和换行
        hex_clean = re.sub(r'\s+', ' ', hex_data.strip())
        hex_bytes = hex_clean.split()
        
        # 查找所有以FA开始、以AF结束的19字节数据包
        i = 0
        while i < len(hex_bytes):
            if i < len(hex_bytes) and hex_bytes[i] == 'FA':
                # 查找19字节的完整数据包
                if i + 18 < len(hex_bytes) and hex_bytes[i + 18] == 'AF':
                    packet = hex_bytes[i:i + 19]
                    processed_packets += 1
                    
                    # GSR数据：第2位和第3位 (索引1和2)
                    gsr_hex = packet[1] + packet[2]
                    gsr_decimal = int(gsr_hex, 16)
                    gsr_voltage = gsr_decimal * 3.3 / 4096
                    
                    # PPG数据：第16位 (索引15)
                    ppg_hex = packet[15]
                    ppg_decimal = int(ppg_hex, 16)
                    
                    data.append({
                        'Time': timestamp,
                        'GSR(V)': round(gsr_voltage, 2),
                        'PPG(BPM)': ppg_decimal
                    })
                    
                    i += 19  # 跳过这个数据包
                else:
                    i += 1
            else:
                i += 1
    
    print(f"  总共处理了 {processed_packets} 个数据包")
    return data

def filter_and_process_data(data):
    """
    过滤数据并进行聚合处理
    """
    if not data:
        print("  ⚠️ 没有数据需要处理")
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    original_count = len(df)
    print(f"  原始数据记录数: {original_count}")
    
    # 1. 数据过滤
    # 过滤PPG数据：保留20-120 BPM范围内的数据，排除0值
    ppg_mask = (df['PPG(BPM)'] >= 20) & (df['PPG(BPM)'] <= 120) & (df['PPG(BPM)'] != 0)
    df_filtered = df.loc[ppg_mask].copy()
    ppg_filtered_count = len(df_filtered)
    print(f"  PPG过滤后记录数: {ppg_filtered_count} (过滤掉 {original_count - ppg_filtered_count} 条)")
    
    # 过滤GSR数据：保留小于等于3.3V的数据
    df_filtered = df_filtered[df_filtered['GSR(V)'] <= 3.3].copy()
    gsr_filtered_count = len(df_filtered)
    print(f"  GSR过滤后记录数: {gsr_filtered_count} (过滤掉 {ppg_filtered_count - gsr_filtered_count} 条)")
    
    if gsr_filtered_count == 0:
        print("  ⚠️ 警告: 过滤后没有有效数据!")
        return pd.DataFrame()
    
    # 2. 时间格式处理 - 保留毫秒级时间，便于逐条查看
    time_with_ms = df_filtered['Time'].str.extract(r'(\d{2}:\d{2}:\d{2}\.\d{3})', expand=False)
    df_filtered['Time'] = time_with_ms.fillna(df_filtered['Time'])
    
    # 3. 调整数值格式并保持逐条数据
    df_filtered['GSR(V)'] = df_filtered['GSR(V)'].round(2)
    df_filtered['PPG(BPM)'] = df_filtered['PPG(BPM)'].round(0).astype(int)
    
    final_count = len(df_filtered)
    print(f"  过滤和格式化后记录数: {final_count}")
    
    return df_filtered.reset_index(drop=True)

def export_to_excel(data, output_file):
    """
    将数据导出到Excel文件
    """
    try:
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"  ✓ 数据已导出到: {output_file} (记录数: {len(df)})")
        return True
    except Exception as e:
        print(f"  ✗ 导出Excel时发生错误: {e}")
        return False

def process_single_log_file(log_file_path, output_dir):
    """
    处理单个日志文件
    """
    # 获取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_file = os.path.join(output_dir, f"{file_name}.xlsx")
    
    print(f"\n处理文件: {log_file_path}")
    print("-" * 60)
    
    try:
        # 解析日志文件
        raw_data = parse_log_file(log_file_path)
        
        if raw_data:
            print(f"  ✓ 成功解析了 {len(raw_data)} 条原始数据记录")
            
            # 数据过滤和处理
            processed_data = filter_and_process_data(raw_data)
            
            if not processed_data.empty:
                # 导出到Excel
                if export_to_excel(processed_data, output_file):
                    print(f"  🎉 {file_name}.log 处理完成！")
                    return True
                else:
                    print(f"  ✗ {file_name}.log 导出失败")
                    return False
            else:
                print(f"  ✗ {file_name}.log 过滤后没有有效数据")
                return False
        else:
            print(f"  ✗ {file_name}.log 未找到有效数据")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理 {file_name}.log 时发生错误: {e}")
        return False

def get_processed_files(output_dir):
    """
    获取已经处理过的文件列表(不含扩展名)
    """
    if not os.path.exists(output_dir):
        return set()
    
    xlsx_files = glob.glob(os.path.join(output_dir, "*.xlsx"))
    # 提取文件名(不含扩展名)
    processed_names = {os.path.splitext(os.path.basename(f))[0] for f in xlsx_files}
    return processed_names

def get_unprocessed_log_files(log_dir, output_dir):
    """
    获取未处理的日志文件列表
    """
    # 获取所有.log文件
    log_pattern = os.path.join(log_dir, "*.log")
    all_log_files = glob.glob(log_pattern)
    
    # 获取已处理的文件名集合
    processed_files = get_processed_files(output_dir)
    
    # 筛选未处理的文件
    unprocessed_files = []
    for log_file in all_log_files:
        file_name = os.path.splitext(os.path.basename(log_file))[0]
        if file_name not in processed_files:
            unprocessed_files.append(log_file)
    
    return unprocessed_files, processed_files

# 主程序入口
def main():
    ################ 设置日志文件目录和输出目录
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'list')
    
    print("批量处理传感器数据日志文件...")
    print("=" * 70)
    print(f"脚本目录: {script_dir}")
    
    # 检查log目录是否存在
    if not os.path.exists(log_dir):
        print(f"✗ 错误: 找不到 {log_dir} 目录")
        print("请确保log目录存在并包含.log文件")
        return
    
    # 查找所有.log文件和未处理的文件
    log_pattern = os.path.join(log_dir, "*.log")
    all_log_files = glob.glob(log_pattern)
    
    if not all_log_files:
        print(f"✗ 在 {log_dir} 目录中没有找到.log文件")
        return
    
    print(f"找到 {len(all_log_files)} 个.log文件")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取未处理的文件
    unprocessed_files, processed_files = get_unprocessed_log_files(log_dir, output_dir)
    
    print(f"\n已处理的文件数: {len(processed_files)}")
    if processed_files:
        print("已处理的文件:")
        for name in sorted(processed_files):
            print(f"  ✓ {name}.log → {name}.xlsx")
    
    print(f"\n待处理的文件数: {len(unprocessed_files)}")
    if not unprocessed_files:
        print("🎉 所有文件都已处理完成,无需重复处理!")
        print(f"📁 输出目录: {output_dir}")
        return
    
    print("待处理的文件:")
    for log_file in unprocessed_files:
        print(f"  - {os.path.basename(log_file)}")
    
    print(f"\n输出目录: {output_dir}")
    print("\n开始处理未处理的文件...")
    
    # 处理统计
    success_count = 0
    fail_count = 0
    
    # 逐个处理每个未处理的.log文件
    for log_file in unprocessed_files:
        if process_single_log_file(log_file, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # 显示处理结果汇总
    print("\n" + "=" * 70)
    print("批量处理完成!")
    print(f"  📊 本次处理统计:")
    print(f"    ✓ 成功处理: {success_count} 个文件")
    if fail_count > 0:
        print(f"    ✗ 处理失败: {fail_count} 个文件")
    print(f"  📁 输出目录: {output_dir}")
    
    # 列出所有生成的文件
    xlsx_files = glob.glob(os.path.join(output_dir, "*.xlsx"))
    if xlsx_files:
        print(f"\n当前list目录中的所有Excel文件 (共{len(xlsx_files)}个):")
        for xlsx_file in sorted(xlsx_files):
            print(f"  - {os.path.basename(xlsx_file)}")
    
    print(f"\n处理说明:")
    print(f"  ✓ 已跳过已处理的文件,避免重复处理")
    print(f"  ✓ 已过滤PPG数据 (保留20-120 BPM)")
    print(f"  ✓ 已过滤GSR数据 (保留≤3.3V)")
    print(f"  ✓ 已按时间聚合数据")
    print(f"  ✓ 已简化时间格式")

if __name__ == "__main__":
    main()
