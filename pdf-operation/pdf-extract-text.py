import PyPDF2
import sys
import re


# 获取命令行参数
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    print("Please provide a file path as a command-line argument.")
    sys.exit()

# 打开 PDF 文件和输出文件
with open(file_path, 'rb') as pdf_file, open('output.txt', 'w', encoding='utf-8') as output_file:
    # 创建 PDF 阅读器对象
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # 循环遍历每一页并提取文本
    for page in pdf_reader.pages:
        text = page.extract_text()

        # 处理提取的文本
        if text:
            # 将文本写入输出文件
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
            text = re.sub(r'(?<=\S)[^\.\?!:]\s+(\S)', r' \1', text)
            output_file.write(text)
            output_file.write('\n') # 每一页的文本之间用空行隔开
