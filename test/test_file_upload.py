import requests

url = 'http://localhost:8888/file'  # 服务器的上传接口地址

# 选择要上传的文件
file_path = 'test/20221001_trade.csv'

# 发送文件数据到服务器
with open(file_path, 'rb') as file:
    response = requests.post(url, files={'file': file})

# 处理服务器的响应
if response.status_code == 200:
    print('文件上传成功！')
else:
    print('文件上传失败！')