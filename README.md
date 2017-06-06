# brt
Board Recognizor of Tiantian 


识别准备分两步走：一是使用Tensor flow训练再识别，但不识别颜色。
颜色使用另外的机制，如simpleCV或openCV来做；

这样做，可以使用灰度图来识别，这样效率应该高。

棋子虽然只有几种，但仍使用MNIST的存储格式，这样便于复用。以备将来处理其它设备、软件的截图。

使用simpleCV截出的棋子图，置于resources目录下。


scaffold 目录下的getChessmen中，介绍了如何使用simpleCV截出棋子的方法。

