

目前使用docker版的simpleCV 来取得棋子。


启动docker


cd /Users/papa/AI/CV/simpleCV/image-classification
start docker:

docker run --rm -v `pwd`:/opt/work -p 54717:8888 -t -i sightmachine/simplecv

这个要使用浏览器访问；所有命令都是在浏览器中运行。


注意要在notebook中，执行 cd /opt/work； 具体方法如下：
浏览器访问： http://localhost:54717/

点New notebook 按钮

在输入中，执行（点页面上的那个播放按钮）：
cd /opt/work

然后执行代码：

from SimpleCV import *
disp = Display(displaytype='notebook')
screenshot = Image('resources/full-board.jpg')
screenshot.save(disp)

这个会显示一个完整的棋盘。

下面的代码，能找出32个图，但找不全子，有一个小企鹅头
len(matches)是32


tmp2 = Image('/opt/work/resources/tmp2.png', sample=True)
matches = screenshot.findTemplate(tmp2, threshold=4.1,method="CCOEFF")
for i in range(32):
  m = matches[i]
  cm = screenshot.crop(m.x, m.y,m.width(), m.height() )
  cm.save(disp)
  
  这个就少一个红士。
  
  
  


