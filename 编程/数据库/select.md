- `from r1,r2,r3...`: 对r1,r2,r3...做笛卡尔积. 其中元素名称相同的会以r1.id, r2.id区分
- `where`:选择符合条件的项
	- `where course.id=takes.id`
	- `where course.id=5`
	- `where course.id in/not in(select ID from teaches where year=2024)`
	- `where name not in ('Mozart', 'Einstein')`
	- `where (course_id, sec_id, semester, year) in (select course_id, sec_id, semester, year from teaches where teaches.ID= '10101')`
- `as`: 重命名
	- 跟在select后, 或是from后都可以, 如`select A as S`,也可以`select A from R1 as R`
	- 用于重命名avg()等函数, 因为函数不会给新属性一个正常名字. 可以`select avg(A) as avg_A`
- `order by`: 对结果排序, 默认递增. 
	- `desc`表示递减, `asc`表示递增
	- `select * from instructor order by salary desc, name asc`
- `group by`: 对结果按组排序, 一组的在一起
	- 配合`avg()`分组求平均: [[聚合函数#^792894]]
- `having`: 选择满足条件的组
	- 必须和`group by`配套使用
- 返回一个RELATION
- 默认不去重, 添加后缀**distinct**去重


一次`select`的执行过程: 
1. `from`: 进行笛卡尔积, 获得临时RELATION
2. `where`: 对每一项施加约束, 选出满足条件的项
3. `group by`: 分组, 如果没有`group by`语句, 视为分成1组
4. `having`: 对每一组施加约束, 选出满足条件的组

常见SELECT操作
1. $\Join$ 的表示
		`select A from r1,r2 where r1.id=r2.id`
		也就是显式的指定了连接的依据 `r1.id=r2.id`
2. 比大小
		`select A from R1 as S,R2 as T where S.SCORE>T.SCORE`
		将自身与自身做笛卡尔积
3. 有全部
		`not exist (select *) except (select .. where)`