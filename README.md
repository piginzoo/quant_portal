# 初衷

想做个超简单的web端，来查看自己的交易记录和市值等信息。
之前在quant_trader里面写过，但是过于个性化，不通用，还复杂。
我也不是web高手，不整啥vue啥的了，jquery+html搞定，画图matlibplot全撸，简单一些，不难为自己。
这次搞个超简单的，只有：
- 支持登录
- 一个页面，不整那么复杂了
- 支持多个账户
- 仅显示市值变化、交易记录、当前账户信息
- 支持文件上传
嗯，够了，不整太复杂，简单可以看下面的miniqmt传上来的交易记录就行。

# 实现细节

第一步，先不考虑多账户了，未来改造也容易。

实现一下功能：
- 认证简单点，还是在服务器上存个密码就行，客户端存个，对称密钥
- 每次文件都推送上去，写一个通用文件推送的接口
- 其实就是每次让服务器上都和本地同步文件
    - 订单文件
    - 交易文件
    - 每日市值文件
- 展示：显示在一页里，省事
    - 当前仓位市值等信息
    - 市值曲线 vs 大盘、中证1000（用matlibplot画svg），不整js chart那种库了
    - 收益率评价
        - 。。。那些统计指标
    - 历史交易记录：显示trade，日期倒序

# 服务器配置

在ubuntu上，安装nginx和python。
转发原因是1024以下端口不允许非root使用，那就转发呗。

修改nginx的配置，加入：

```
        upstream flask {
                server 127.0.0.1:8080;
        }

        server {
               listen 80;
               server_name <你的域名>;
               client_max_body_size 8M;
               client_body_buffer_size 2M;
               location / {
                   proxy_pass_header Server;
                   proxy_set_header Host $http_host;
                   proxy_redirect off;
                   proxy_set_header X-Real-IP $remote_addr;
                   proxy_set_header X-Scheme $scheme;
                   proxy_http_version 1.1;
                   proxy_set_header Upgrade $http_upgrade;
                   proxy_set_header Connection "Upgrade";
                   proxy_pass http://flask;
                }
        }
```

这样，就不用做转发了。

# matplotlib中文乱码问题解决

参考：
- https://blog.csdn.net/takedachia/article/details/131017286
- https://juejin.cn/post/7023987001275711496

```
git clone https://github.com/tracyone/program_font 
cd program_font 
./install.sh
ll /usr/share/fonts/MyFonts

python
>>> import matplotlib
>>> matplotlib.matplotlib_fname()
'/home/ubuntu/.local/lib/python3.10/site-packages/matplotlib/mpl-data/matplotlibrc'

cp /usr/share/fonts/MyFonts/simhei.ttf  /home/ubuntu/.local/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/

rm -rf ~/.cache/matplotlib/*

vim  /home/ubuntu/.local/lib/python3.10/site-packages/matplotlib/mpl-data/matplotlibrc
>>> font.family:  sans-serif
>>> font.serif:      simhei, ......
>>> font.sans-serif: simhei, ......
```

