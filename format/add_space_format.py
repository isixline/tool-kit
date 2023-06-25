import re
import sys

def add_space_between_characters(input_file_path, output_file_path=None):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 在中文和英文之间添加空格
    pattern = r'([\u4e00-\u9fa5])([a-zA-Z])'
    result = re.sub(pattern, r'\1 \2', content)

    # 在英文和中文之间添加空格
    pattern = r'([a-zA-Z])([\u4e00-\u9fa5])'
    result = re.sub(pattern, r'\1 \2', result)

    # 决定写入原始文件还是输出到新文件
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(result)
        print("空格添加完成！输出文件路径:", output_file_path)
    else:
        with open(input_file_path, 'w', encoding='utf-8') as file:
            file.write(result)
        print("空格添加完成！原始文件已被覆盖。")

# 获取命令行参数
args = sys.argv
input_file_path = args[1]  # 第一个参数是输入文件路径
output_file_path = args[2] if len(args) > 2 else None  # 第二个参数是输出文件路径（可选）

# 调用函数进行处理
add_space_between_characters(input_file_path, output_file_path)
