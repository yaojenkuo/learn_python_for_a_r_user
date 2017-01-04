# [第 27 天] 深度學習 TensorFlow

---

在過去幾天我們與功能強大的機器學習套件 **scikit-learn** 相處得還算融洽，不難發現我們對於 **scikit-learn** 的認識其實僅止於冰山一角，它有著包羅萬象的機器學習演算法，資料預處理與演算法績效評估的功能，除了參考[首頁](http://scikit-learn.org/stable/index.html)開宗明義的六大模組，我們也可以參考 **scikit-learn** 官方網站的[機器學習地圖](http://scikit-learn.org/stable/tutorial/machine_learning_map/)：

![day2701](https://storage.googleapis.com/2017_ithome_ironman/day2701.png)

即便 **scikit-learn** 已近乎包山包海，但挑剔的使用者還是在雞蛋裡挑骨頭，近年的當紅炸子雞深度學習（Deep learning）在哪裡？這對於連續閱讀 **scikit-learn** 文件好幾天的我們無疑是個好消息，暫時可以換個口味了！

目前主流的深度學習框架（Framework）有 **Caffe**、**TensorFlow**、**Theano**、**Torch** 與 **Keras**，其中 **Keras** 是可以使用 API 呼叫方式同時使用 **TensorFlow** 與 **Theano** 的高階框架，我們選擇入門的框架是 **TensorFlow**。

## 安裝 TensorFlow

我們的開發環境是 [Anaconda](https://www.continuum.io/downloads)，如果你對本系列文章的開發環境有興趣，可以參照 [[第 01 天] 建立開發環境與計算機應用](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day01.md)。

### 第一步（pip 安裝）

在終端機使用 `pip` 指令安裝 **TensorFlow**：

```
$ pip install tensorflow
```

更多的安裝細節（GPU版本或其他作業系統...等），請參考官方的[安裝指南](https://www.tensorflow.org/get_started/os_setup)。

### 第二步（測試）

進入 jupyter notebook 測試以下程式：

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

![day2702](https://storage.googleapis.com/2017_ithome_ironman/day2702.png)

## 快速實作

讓我們跟著官方文件實作第一個 **TensorFlow** 程式，我們要利用梯度遞減（Gradient descent）的演算法找出已知迴歸模型（y = 0.1x + 0.3）的係數（0.1）與截距（0.3）並對照結果。

```python
import tensorflow as tf
import numpy as np

# 準備資料
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# W 指的是係數，斜率介於 -1 至 1 之間
# b 指的是截距，從 0 開始逼近任意數字
# y 指的是預測值
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 我們的目標是要讓 loss（MSE）最小化
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()

# 將神經網絡圖畫出來
sess = tf.Session()
sess.run(init)

# 將迴歸線的係數與截距模擬出來
# 每跑 20 次把當時的係數與截距印出來
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        
# 關閉 Session
sess.close()
```

![day2703](https://storage.googleapis.com/2017_ithome_ironman/day2703.png)

**TensorFlow** 一直到 `sess.run()` 才開始作用，前面都只是在建立資料流的結構。

## 基礎使用

應用 **TensorFlow** 的時候我們得了解她的名詞與定義：

|名詞|定義|
|---|---|
|Graphs|建立運算元|
|Sessions|執行運算|
|Tensors|資料|
|Variables|變數|
|Feeds|資料輸入|
|Fetches|資料輸出|

### 建立運算元

```python
import tensorflow as tf

# 1x2 的矩陣
matrix1 = tf.constant([[3, 3]])

# 2x1 的矩陣
matrix2 = tf.constant([[2],
                       [2]]
                      )

# matmul() 方法代表是矩陣的乘法，答案是 12
product = tf.matmul(matrix1, matrix2)
```

![day2704](https://storage.googleapis.com/2017_ithome_ironman/day2704.png)

現在我們的運算元已經建立好，有三個節點，分別是兩個 `constant()` 與一個 `matmul()`，意即神經網絡的圖已經建構完成，但是尚未執行運算。

### 執行運算

#### 方法一

記得使用 `close()` 方法關閉 Session。

```python
import tensorflow as tf

# 1x2 的矩陣
matrix1 = tf.constant([[3, 3]])

# 2x1 的矩陣
matrix2 = tf.constant([[2],
                       [2]]
                      )

# matmul() 方法代表是矩陣的乘法
product = tf.matmul(matrix1, matrix2)

# 啟動 Session
sess = tf.Session()
result = sess.run(product)
print(result)

# 關閉 Session
sess.close()
```

![day2705](https://storage.googleapis.com/2017_ithome_ironman/day2705.png)

#### 方法二

不需要另外關閉 Session。

```python
import tensorflow as tf

# 1x2 的矩陣
matrix1 = tf.constant([[3, 3]])

# 2x1 的矩陣
matrix2 = tf.constant([[2],
                       [2]]
                      )

# matmul() 方法代表是矩陣的乘法
product = tf.matmul(matrix1, matrix2)

# 啟動
with tf.Session() as sess:
    result = sess.run([product])
    print(result)
```

![day2707](https://storage.googleapis.com/2017_ithome_ironman/day2707.png)

#### 方法三

要將 `matrix1` 設定為 `Variable` 然後再由她來初始化。

```python
# 啟動 Session
import tensorflow as tf
sess = tf.InteractiveSession()

# 1x2 的矩陣
# 注意這裡改變成 Variable
matrix1 = tf.Variable([[3, 3]])

# 2x1 的矩陣
matrix2 = tf.constant([[2],
                       [2]]
                      )

# 初始化 `matrix1`
matrix1.initializer.run()

# 執行運算
result = tf.matmul(matrix1, matrix2)
print(result.eval())

# 關閉 Session
sess.close()
```

![day2708](https://storage.googleapis.com/2017_ithome_ironman/day2708.png)

### 變數

```python
import tensorflow as tf

# 建立 Variable
state = tf.Variable(0, name="counter")

# 每次加 1 之後更新 state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 初始化
init_op = tf.global_variables_initializer()

# 執行運算
with tf.Session() as sess:
    sess.run(init_op)
    # 印初始值
    print(sess.run(state))
    # 更新三次分別印出 Variable
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        
# 關閉 Session
sess.close()
```

![day2709](https://storage.googleapis.com/2017_ithome_ironman/day2709.png)

### 資料輸入

先利用 `tf.placeholder()` 宣告資料的種類，在執行的時候才將資料以字典（dictionary）的結構輸入。

```python
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

# 將 input1 以 7 輸入，input2 以 3 輸入
with tf.Session() as sess:
    print(sess.run([output], feed_dict = {input1: [7], input2: [3]}))
```

![day2710](https://storage.googleapis.com/2017_ithome_ironman/day2710.png)

### 資料輸出

```python
input1 = tf.constant([3])
input2 = tf.constant([5])
added = tf.add(input1, input2)
multiplied = tf.mul(input1, input2)

# 輸出 added 與 multiplied
with tf.Session() as sess:
    result = sess.run([added, multiplied])
    print(result)
```

![day2711](https://storage.googleapis.com/2017_ithome_ironman/day2711.png)

## 小結

第二十七天我們開始練習使用 Python 的神經網絡套件 **TensorFlow**，我們成功安裝了 `tensorflow`，並實作官方文件的第一個練習：使用梯度遞減逼近迴歸模型的係數與截距。此外我們也從基礎使用的範例中瞭解 **TensorFlow** 模型產出的過程，我們首先建立運算元，畫出神經網絡圖，初始化變數，最後才是執行運算並輸出結果，這也跟我們在前面幾天的練習中，先初始化一個分類器然後才將資料投入進行運算的概念相似。

## 參考連結

- [Introduction - TensorFlow](https://www.tensorflow.org/get_started/)