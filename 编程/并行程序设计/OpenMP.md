- OpenMP is a multi-threading, shared address model.
	- Threads communicate by **sharing variables**. 
- Unintended sharing of data causes race conditions
	- race condition: when the program’s outcome changes as the threads are scheduled differently. 
- To control race conditions:
	-  Use synchronization to protect data conflicts. 
- Synchronization is expensive so:
	- Change how data is accessed to minimize the need for synchronization.

openMP使用[[directive]]来设置[[parallel region]].