Context
- constant: pre中的元素的声明
- axiom: pre中的约束

Machine
不管程序执行的具体内容, 只描述需求,
- variable
	- post中的元素的声明
- invariant
	- post中元素的约束, 比如i$\in$N
- event
	1. final事件, 即程序结束
		- guard设置为post
			- 一旦满足guard, 则意味着程序可以实现post
	2. progress事件, 代表程序执行
		- 设置为anticipated, 代表它会成立, 但是现在不知道具体是怎么做的
		- action中包含 #todo 没懂
	3. init
		- 赋值, 可以用  i :$\in$ N的形式, 表示赋了某一个自然数

第二步
refined Machine
- invariant
	- 不变式中的内容
- final
	- guard改成$\neg$P
- event
	- init
		- 变量的初值
	- progress
		- anticipated 改成convergent, 意味着它是实现的功能
		- action 程序中的内容
		- guard 加上P

convergent: rodin会检查变式, 确保progress会终止
