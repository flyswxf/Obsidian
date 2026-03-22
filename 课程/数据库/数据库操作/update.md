- `r`: 表名
- `set column = value`: 更新的数据
- `where`: 选择符合条件的项
	- 如果不加where, 会更新表中所有的项


```
UPDATE Websites 
SET alexa='5000', country='USA' 
WHERE name='菜鸟教程';
```