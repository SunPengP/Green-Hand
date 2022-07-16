- Image.open打开图片，使用shape是出错

![image-20220401173621971](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220401173621971.png)

原因：Image.open()的结果是RGB图像数据但是程序里用的是RGB数组，所以报错。 cv2.imread()的结果则是直接的BGR数组。

```python
解决方法：使用cv2
import cv2
img=cv2.imread('C:/Users/lenovo/Desktop/anchor.jpg')
w,h = img.shape[0],img.shape[1]
```

