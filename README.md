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