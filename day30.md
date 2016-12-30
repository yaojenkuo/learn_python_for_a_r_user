# [第 30 天] 深度學習（4）卷積神經網絡與鐵人賽總結

---

我們今天會練習使用神經網絡的套件 **TensorFlow** 來建立我們的第一個深度學習模型：卷積神經網絡（Convolutional Neural Network，CNN），來提升原本 **92%** 準確率的 **MNIST** 手寫數字辨識模型。卷積神經網絡廣泛被運用在圖片處理領域，我們很快地簡介她的特性。

## 卷積神經網絡

卷積神經網絡要解決的問題源自於使用神經網絡辨識高解析度的彩色圖片，馬上會遭遇運算效能不足的難題，卷積神經網絡使用兩個方法來解決這個難題：

### Convolution

有些特徵不需要看整張圖片才能捕捉起來，為了解決運算效能不足而採取將圖片解析度降維的方法，使用 Sub-sampling 或 Down-sampling 稱呼也許可以讓我們更容易瞭解她的意義。

![day3001](https://storage.googleapis.com/2017_ithome_ironman/day3001.gif)
Source: [Artificial Inteligence](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolution.html)

### Max-pooling

這是為了確保經過 Convolution 後圖片中的特徵可以被確實保留下來而採取的方法。

![day3002](https://storage.googleapis.com/2017_ithome_ironman/day3002.jpeg)
Source: [What is max pooling in convolutional neural networks?](https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks)

今天實作的深度練習模型除了導入 **Convolution** 與 **Max-pooling** 以外，還有三個地方與[昨天](http://ithelp.ithome.com.tw/articles/10187912)不同：

### 不同的 Activation Function

先前在添加神經網絡層的時候，如果沒有指定 **Activation function** 就是使用預設的線性函數，但是在 **CNN** 中會使用 ReLU（Rectified Linear Unit）作為 **Activation function**，模擬出非線性函數，確保神經元輸出的值在 0 到 1 之間。

![day3004](https://storage.googleapis.com/2017_ithome_ironman/day3004.png)
Source: [Rectifier (neural networks) - Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

### 新增 Dropout 函數

用來避免過度配適（Overfitting）。

> To reduce overfitting, we will apply [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) before the readout layer.
> [Deep MNIST for Experts | TensorFlow](https://www.tensorflow.org/tutorials/mnist/pros/)

### 更換 Optimizer

將我們先前一直使用的梯度遞減（Gradient descent）更換為 **ADAM**，是一個更進階且更成熟的梯度遞減演算法。

> We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.
> [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

## TensorFlow 實作

### 卷積神經網絡架構

我們來拆解一下接著要建立的卷積神經網絡架構：

- 輸入圖片（解析度 28x28 的手寫數字圖片）
- 第一層是 Convolution 層（32 個神經元），會利用解析度 5x5 的 filter 取出 32 個特徵，然後將圖片降維成解析度 14x14
- 第二層是 Convolution 層（64 個神經元），會利用解析度 5x5 的 filter 取出 64 個特徵，然後將圖片降維成解析度 7x7
- 第三層是 Densely Connected 層（1024 個神經元），會將圖片的 1024 個特徵攤平
- 輸出結果之前使用 Dropout 函數避免過度配適
- 第四層是輸出層（10 個神經元），使用跟之前相同的 Softmax 函數輸出結果

### 實作

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 設定參數
logs_path = 'TensorBoard/'
n_features = x_train.shape[1]
n_labels = y_train.shape[1]

# 啟動 InteractiveSession
sess = tf.InteractiveSession()
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, n_features])
with tf.name_scope('Label'):
    y_ = tf.placeholder(tf.float32, shape=[None, n_labels])

# 自訂初始化權重的函數
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
# 自訂 convolution 與 max-pooling 的函數
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一層是 Convolution 層（32 個神經元），會利用解析度 5x5 的 filter 取出 32 個特徵，然後將圖片降維成解析度 14x14
with tf.name_scope('FirstConvolutionLayer'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# 第二層是 Convolution 層（64 個神經元），會利用解析度 5x5 的 filter 取出 64 個特徵，然後將圖片降維成解析度 7x7
with tf.name_scope('SecondConvolutionLayer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# 第三層是 Densely Connected 層（1024 個神經元），會將圖片的 1024 個特徵攤平
with tf.name_scope('DenselyConnectedLayer'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 輸出結果之前使用 Dropout 函數避免過度配適
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四層是輸出層（10 個神經元），使用跟之前相同的 Softmax 函數輸出結果
with tf.name_scope('ReadoutLayer'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 訓練與模型評估
with tf.name_scope('CrossEntropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    tf.summary.scalar("CrossEntropy", cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

# 初始化
sess.run(tf.global_variables_initializer())

# 將視覺化輸出
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        summary = sess.run(merged, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(summary, i)
        writer.flush()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict = {x: x_test, y_: y_test, keep_prob: 1.0}))

# 關閉 session
sess.close()

# 這段程式大概需要跑 30 到 60 分鐘不等，依據電腦效能而定
```

![day3003](https://storage.googleapis.com/2017_ithome_ironman/day3003.png)

![day3005](https://storage.googleapis.com/2017_ithome_ironman/day3005.png)

![day3006](https://storage.googleapis.com/2017_ithome_ironman/day3006.png)

## 小結

第三十天我們繼續練習 Python 的神經網絡套件 **TensorFlow**，針對 **MNIST** 資料建立了第一個深度學習模型：卷積神經網絡，達到 **99%** 左右的準確率。

## 參考連結

- [Deep MNIST for Experts | TensorFlow](https://www.tensorflow.org/tutorials/mnist/pros/)
- [[DSC 2016] 系列活動：李宏毅 / 一天搞懂深度學習](http://www.slideshare.net/tw_dsconf/ss-62245351)
- [Hvass-Labs@GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb)
- [c1mone@GitHub](https://github.com/c1mone/Tensorflow-101/blob/master/notebooks/Ch1.2_MNIST_Convolutional_Network.ipynb)
- [Start on TensorBoard](http://robromijnders.github.io/tensorflow_basic/)

## 鐵人賽總結

### 學習筆記的脈絡

這份學習筆記從一個 R 語言使用者學習 Python 在資料科學的應用，並且相互對照的角度出發，整份學習筆記可以分為五大主題：

#### 基礎

- [[第 01 天] 建立開發環境與計算機應用](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day01.md)
- [[第 02 天] 基本變數類型](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day02.md)
- [[第 03 天] 變數類型的轉換](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day03.md)
- [[第 04 天] 資料結構 List，Tuple 與 Dictionary](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day04.md)
- [[第 05 天] 資料結構（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day05.md)
- [[第 06 天] 資料結構（3）Data Frame](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day06.md)
- [[第 07 天] 迴圈與流程控制](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day07.md)
- [[第 08 天] 函數](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day08.md)
- [[第 09 天] 函數（2）](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day09.md)
- [[第 10 天] 物件導向 R 語言](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day10.md)
- [[第 11 天] 物件導向（2）Python](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day11.md)

#### 基礎應用

- [[第 12 天] 常用屬性或方法 變數與基本資料結構](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day12.md)
- [[第 13 天] 常用屬性或方法（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day13.md)
- [[第 14 天] 常用屬性或方法（3）Data Frame](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day14.md)
- [[第 15 天] 載入資料](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day15.md)
- [[第 16 天] 網頁解析](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day16.md)
- [[第 17 天] 資料角力](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day17.md)

#### 視覺化

- [[第 18 天] 資料視覺化 matplotlib](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day18.md)
- [[第 19 天] 資料視覺化（2）Seaborn](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day19.md)
- [[第 20 天] 資料視覺化（3）Bokeh](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day20.md) 

#### 機器學習

- [[第 21 天] 機器學習 玩具資料與線性迴歸](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day21.md)
- [[第 22 天] 機器學習（2）複迴歸與 Logistic 迴歸](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day22.md)
- [[第 23 天] 機器學習（3）決策樹與 k-NN 分類器](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day23.md)
- [[第 24 天] 機器學習（4）分群演算法](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day24.md)
- [[第 25 天] 機器學習（5）整體學習](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day25.md)
- [[第 26 天] 機器學習（6）隨機森林與支持向量機](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day26.md)

#### 深度學習

- [[第 27 天] 深度學習 TensorFlow](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day27.md)
- [[第 28 天] 深度學習（2）TensorBoard](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day28.md)
- [[第 29 天] 深度學習（3）MNIST 手寫數字辨識](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day29.md)
- [[第 30 天] 深度學習（4）卷積神經網絡與鐵人賽總結](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day30.md)

### 主觀比較表

我們還是不免落於俗套地依據學習筆記由淺入深的脈絡，整理兩個語言在各個面向的主觀比較表（純為個人意見，理性勿戰）：

面向|R 語言|Python|原因
---|---|---|---
開發環境|和|和|兩個程式語言的安裝都相當簡單，也都有絕佳的 IDE 可供使用。
變數|和|和|兩個程式語言在變數的指派，判斷與轉換上都很直覺。
資料結構|勝|負|R 語言內建資料框與 `element-wise` 運算，Python 需要仰賴 `pandas` 與 `numpy`。
資料載入|和|和|兩個程式語言都可以透過套件的支援載入表格式資料或 JSON 等資料格式。
網頁解析|和|和|R 語言的 `rvest` 與 Python 的 `BeautifulSoup` 都能讓使用者輕鬆解析網頁資料。
資料角力|和|和|兩個程式語言的套件或函數都支援常見的資料角力技巧。
玩具資料|勝|負|R 語言內建的玩具資料數量多，文件說明豐富。
資料視覺化|勝|負|R 語言 `ggplot2` 的多樣繪圖功能搭配 `ggplotly()`，可以更簡單地畫出美觀的圖。
機器學習（基礎）|勝|負|R 語言內建 `stats` 套件中的 `lm()`，`kmeans()` 與 `glm()` 可以快速建立基礎的機器學習模型。
機器學習（進階）|負|勝|Python 的 `scikit-learn` 將所有的機器學習演算法都整理成一致的方法供使用者呼叫。
深度學習|負|勝|目前主流的神經網絡框架，TensorFlow，Theano 與高階的 Keras 主要都是使用 Python 實作。

### 主觀判斷

我們也不免落於俗套地除了前述的主觀比較表，也針對一些使用者的特徵提供主觀判斷：

使用者特徵|建議
---|---
我沒有寫過程式|R 語言
我喜歡函數型編程|R 語言
我喜歡物件導向編程|Python
我想要作統計分析為主|R 語言
我想要作資料視覺化為主|R 語言
我想要建置深度學習模型為主|Python
我想要在網站後端建置機器學習系統為主|Python

### 學習資源整理

我們也不免落於俗套地整理了自學的書籍或網站：

#### R 語言

- [R in Action](https://www.manning.com/books/r-in-action-second-edition)
- [R Cookbook](http://shop.oreilly.com/product/9780596809164.do)
- [The Art of R Programming](https://www.amazon.com/Art-Programming-Statistical-Software-Design/dp/1593273843)
- [Advanced R](https://www.amazon.com/Advanced-Chapman-Hall-Hadley-Wickham/dp/1466586966)
- [R Graphics Cookbook](http://shop.oreilly.com/product/0636920023135.do)
- [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
- [DataCamp](https://www.datacamp.com)

#### Python

- [Codecademy](https://www.codecademy.com)
- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [Learn Python the Hard Way](https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0321884914)
- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)
- [scikit-learn - Machine Learning in Python](http://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)
- [DataCamp](https://www.datacamp.com)

### 初衷

初次聽到 iT 邦幫忙鐵人賽是在自學 [Git](https://git-scm.com/) 的時候看保哥 [30 天精通 Git 版本控管](https://github.com/doggy8088/Learn-Git-in-30-days)，當時就暗自下定決心有機會也要參賽，用 30 天向世界宣告這一年鋼鐵般的鍛鍊！

這份學習筆記野心很大，初衷是希望可以解決一個常見問題：**想要從事資料科學相關的工作，時間只能夠在 R 語言與 Python 中擇一學習，請問各位大大推薦先從哪一個開始？**

所以我們在絕大多數的章節中，讓兩個程式語言處理同一個資料科學問題，並陳她們的語法以便讓讀者可以體會箇中異同，進而達成目標：**讀完我（或我的部分章節），你的心中會自然作出選擇**，請放心且大膽地選擇一個開始你的資料科學旅程，她們都是在資料科學應用上我們能大力仰賴的程式語言。

> Both languages have a lot of similarities in syntax and approach, and you can’t go wrong with one, the other, or both.
> [Vik Paruchuri](https://www.dataquest.io/blog/python-vs-r/)

這是我第一次參加 iT 邦幫忙鐵人賽，很開心也很有成就感能完賽，我們 2018 鐵人賽再見！