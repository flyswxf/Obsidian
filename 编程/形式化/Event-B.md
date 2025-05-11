Context
- constant: pre中的元素的声明
- axiom: pre中的约束

Machine
- variable: post中的元素的声明
- invariant: 
	- post中元素的约束, 比如i$\in$N
- event: 
	1. 必须包含final事件, 即程序结束
		- guard中包含post中的条件, 一旦满足guard, 则意味着程序可以实现post
	2. 必须包含progress事件, 代表程序执行
		- 设置为anticipated, 代表它会成立, 但是现在不知道具体是怎么做的
		- action中包含 #todo 没懂

第二步
refine
- invariant:
	- 不变式中的内容
- final:
	- guard改成Pre,$\neg$P,Q
- progress:
	- anticipated 改成convergent, 意味着它是实现的功能
	- action按程序中的内容
	- guard加上P

convergent: rodin会检查变式, 确保progress会终止
