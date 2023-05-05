# 出力をファイルに保存する
with open('output.txt', 'w') as f:
    print('Hello, world!', file=f)
    print('This is a test.', file=f)
    print('Goodbye!', file=f)