1、表格

| 这个作业属于哪个课程 | [信安1912-软件工程 (广东工业大学 - 计算机学院)](https://edu.cnblogs.com/campus/gdgy/InformationSecurity1912-Softwareengineering/) |
| -------------------- | ------------------------------------------------------------ |
| 这个作业要求在哪里   | [个人项目作业](https://edu.cnblogs.com/campus/gdgy/InformationSecurity1912-Softwareengineering/homework/12146) |
| 这个作业的目标       | [编程]()                                                     |

2、计算模块接口的设计与实现过程

想要算出第二篇论文是否抄袭了第一篇论文，并且计算出抄袭的程度，试着写一个程序来实现，上网搜资料后知道，可以用这样的一个方法：把论文变成一种叫做词嵌入的格式，然后再用神经网络再把词嵌入变成特征向量，但是用了很多次神经网络之后，向量就会变得太大太复杂，所以用几次之后就要简化一次特征向量，经过几轮重复之后，就可以用计算两个向量的相似度，也就是题目要的抄袭程度，重复率

流程图

![zhanghongtu](zhanghongtu.png)

实现过程

按照下面这样的命令依次输入和回车，就可以算出老师给的五篇论文的重复率

![47d763d708abb648b6a20978b1bb1ef](47d763d708abb648b6a20978b1bb1ef.png)

![9a334d35c1a1a39660aacc8e88f3e1d](9a334d35c1a1a39660aacc8e88f3e1d.png)

![a1633743dc41b9a8565003632e52ea3](a1633743dc41b9a8565003632e52ea3.png)

![b6c108a19aeced27aac40fd0233f759](b6c108a19aeced27aac40fd0233f759.png)

![8f4f3f28bccd7a418a57954080e741a](8f4f3f28bccd7a418a57954080e741a.png)

![5ae0a114f2ed7a76637a2c636f022e4](5ae0a114f2ed7a76637a2c636f022e4.png)

![db856b0ecbb5d96039e2da42d0e153b](db856b0ecbb5d96039e2da42d0e153b.png)

3、计算模块接口部分的性能改进

性能分析

![202eddc0600111fd3a2ed2ce90237ba](202eddc0600111fd3a2ed2ce90237ba.png)

![814b73757a0cd0f2bd3fa50f31e8af5](814b73757a0cd0f2bd3fa50f31e8af5.png)

消耗最大的函数

![aa7d01086ba96254b5d864b32701d0d](aa7d01086ba96254b5d864b32701d0d.png)

![b5988391dab78a19ca0598f485aa06a](b5988391dab78a19ca0598f485aa06a.png)

4、计算模块部分单元测试展示

输出覆盖率

![f2f28360418b64433430ee8a62e861b](f2f28360418b64433430ee8a62e861b.png)

![9858a36dba7d912e72bd7f84c9a6be1](9858a36dba7d912e72bd7f84c9a6be1.png)

5、计算模块部分异常处理说明

如果文件名输入错了，就会报错说没有该文件存在，把文件名确认一次就好了

![5216d4e9896bdb0eb5395fba2e641a7](5216d4e9896bdb0eb5395fba2e641a7.png)

6、PSP表格

