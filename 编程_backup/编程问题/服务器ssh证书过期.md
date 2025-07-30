## 错误提示词:
## err_cert_date_invalid


1. 申请新的ssh证书
   ![[个人测试证书管理.png]]
2. 申请后可以直接点击`部署`
	1. 选中云服务器ECS
	2. 创建任务名称(如update_ssl)
	3. 选择ssl证书和云服务器
	4. 填写服务器中**放置ssl文件的位置**: /root/poemServer/ssl
	   #注意 这是绝对路径, 需要包含/root/这个根目录文件夹
	   ![[ssl-certificate-key-path-configuration.png]]
	5. 如果不直接部署, 也可以点击`下载`
		1. 选择pem/key类型的证书格式
		   ![[ssl-证书下载指南.png]]
		2. 将文件下载到本地主机, 解压出`.key, .pem`文件
		3. 通过远程连接**workbench**
			1. 新文件管理
			   ![[workbench文件管理操作.png]]
			2. 找到**放置ssl文件的位置**, 上传文件即可
			   ![[upload-ssl-file-configuration.png]]
	   