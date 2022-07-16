- Pycharm批量替换某个变量：Ctrl+R

类别为背景的锚框通常称为负类锚框，其余则被称为正类锚框

![image-20220403170930703](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220403170930703.png)

### 关于使用不同的方式打开图片时，图片的形状

```python
PIL.Image.open() # 对应形状为RGB
cv2.imread() # 对应形状BGR
# 相互转换
#1.Image对象->cv2(np.adarray)
img = Image.open(path)
img_array = np.array(img)
#2.cv2(np.adarray)->Image对象
img = cv2.imread(path)
img_Image = Image.fromarray(np.uint8(img))
```

