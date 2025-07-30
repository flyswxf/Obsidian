相当于一个临时的表r, 数据库执行带with的语句时, 会将r替换为对应的select语句
#### 用法
`with RA as (select ...), RB as (select ...) select C from RA`