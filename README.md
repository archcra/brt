# brt
Board Recognizor of Tiantian 


识别准备分两步走：一是使用Tensor flow训练再识别，但不识别颜色。
颜色使用另外的机制，如simpleCV或openCV来做；

这样做，可以使用灰度图来识别，这样效率应该高。

棋子虽然只有几种，但仍使用MNIST的存储格式，这样便于复用。以备将来处理其它设备、软件的截图。

使用simpleCV截出的棋子图，置于resources目录下。


scaffold 目录下的getChessmen中，介绍了如何使用simpleCV截出棋子的方法。

一个手机截屏的棋盘图例如下：

![天天的象棋盘](/resources/full-board.jpg)

我们的目标，就是上传一个图片，返回一个fen串。如完整的棋盘，返回这样的串：

rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
