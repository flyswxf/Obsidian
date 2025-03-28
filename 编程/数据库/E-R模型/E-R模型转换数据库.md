强实体: 直接转换
#todo 放张图
弱实体: 把区分属性(discriminator)和对应强实体集主键作为自己的属性


属性->表设计
- 只保留叶子节点
#todo ppt中有 24页
- 多值属性->新的表,包含(ID, Attr), ID和Attr共同作为表的主键
- 主表与多值属性的表增加外键依赖
- 派生属性不要

关系集->表
- Many-To-Many, 表的主键为双方的主键
- Many-To-One, 将One的主键直接加入Many的主键
- One-To-One,同Many-To-One