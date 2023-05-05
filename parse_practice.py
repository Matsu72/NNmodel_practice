import argparse

# 引数の定義
parser = argparse.ArgumentParser(description='This is a program for processing data.')
parser.add_argument('-i', '--input', help='Input file path')
parser.add_argument('-o', '--output', help='Output file path')
parser.add_argument('-t', '--type', choices=['csv', 'json'], default='csv', help='File type')
parser.add_argument('-s', '--string', type=str, help='Input string')

# 引数の解析
args = parser.parse_args()

# 引数の使用
print('Input file:', args.input)
print('Output file:', args.output)
print('File type:', args.type)
print('File type:', args.string)

