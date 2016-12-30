# [第 28 天] 深度學習（2）TensorBoard

---

我們今天繼續練習神經網絡的套件 **TensorFlow**，在[昨天](http://ithelp.ithome.com.tw/articles/10187702)的第一個實作中我們建立一個很單純的神經網絡，利用梯度遞減（Gradient descent）的演算法去逼近線性迴歸模型的係數與截距，但是我們很快就有了疑問：一直提到的建立運算元（Graphs）究竟在哪裡？是看得到的嗎？答案是可以的，我們可以利用 **TensorBoard** 來視覺化神經網絡。

> The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs, we've included a suite of visualization tools called TensorBoard. You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.
> [TensorBoard: Visualizing Learning | TensorFlow](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)

## 整理程式

在視覺化之前，我們先用較模組化的寫法：[MorvanZhou@GitHub](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf15_tensorboard/full_code.py) 的範例程式改寫[昨天](http://ithelp.ithome.com.tw/articles/10187702)的程式。

### 改寫架構

- 定義一個添加層的函數：`add_layer()`
- 準備資料（Inputs）
- 建立 Feeds（使用 `tf.placeholder()` 方法）來傳入資料
- 添加隱藏層與輸出層
- 定義 `loss` 與要使用的 Optimizer（使用梯度遞減）
- 初始化 Graph 並開始運算

### 改寫後的程式

```python
import tensorflow as tf
import numpy as np

# 定義一個添加層的函數
def add_layer(inputs, input_tensors, output_tensors, activation_function = None):
    W = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    b = tf.Variable(tf.zeros([1, output_tensors]))
    formula = tf.add(tf.matmul(inputs, W), b)
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs

# 準備資料
x_data = np.random.rand(100)
x_data = x_data.reshape(len(x_data), 1)
y_data = x_data * 0.1 + 0.3

# 建立 Feeds
x_feeds = tf.placeholder(tf.float32, shape = [None, 1])
y_feeds = tf.placeholder(tf.float32, shape = [None, 1])

# 添加 1 個隱藏層
hidden_layer = add_layer(x_feeds, input_tensors = 1, output_tensors = 10, activation_function = None)

# 添加 1 個輸出層
output_layer = add_layer(hidden_layer, input_tensors = 10, output_tensors = 1, activation_function = None)

# 定義 `loss` 與要使用的 Optimizer
loss = tf.reduce_mean(tf.square(y_feeds - output_layer))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

# 初始化 Graph 並開始運算
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train, feed_dict = {x_feeds: x_data, y_feeds: y_data})
    if step % 20 == 0:
        print(sess.run(loss, feed_dict = {x_feeds: x_data, y_feeds: y_data}))

sess.close()
```

![day2801](https://storage.googleapis.com/2017_ithome_ironman/day2801.png)

我們可以看到隨著每次運算，`loss` 的數值都在降低，表示類似模型不斷在逼近真實模型。

## 視覺化

接著我們要在模組化的程式中使用 `with tf.name_scope():` 為每個運算元命名，然後在神經網絡運算初始之後，利用 `tf.summary.FileWriter()` 將視覺化檔案輸出。

```python
import tensorflow as tf
import numpy as np

# 定義一個添加層的函數
def add_layer(inputs, input_tensors, output_tensors, activation_function = None):
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
        with tf.name_scope('Biases'):
            b = tf.Variable(tf.zeros([1, output_tensors]))
        with tf.name_scope('Formula'):
            formula = tf.add(tf.matmul(inputs, W), b)
        if activation_function is None:
            outputs = formula
        else:
            outputs = activation_function(formula)
        return outputs

# 準備資料
x_data = np.random.rand(100)
x_data = x_data.reshape(len(x_data), 1)
y_data = x_data * 0.1 + 0.3

# 建立 Feeds
with tf.name_scope('Inputs'):
    x_feeds = tf.placeholder(tf.float32, shape = [None, 1])
    y_feeds = tf.placeholder(tf.float32, shape = [None, 1])

# 添加 1 個隱藏層
hidden_layer = add_layer(x_feeds, input_tensors = 1, output_tensors = 10, activation_function = None)

# 添加 1 個輸出層
output_layer = add_layer(hidden_layer, input_tensors = 10, output_tensors = 1, activation_function = None)

# 定義 `loss` 與要使用的 Optimizer
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(y_feeds - output_layer))
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss)

# 初始化 Graph
init = tf.global_variables_initializer()
sess = tf.Session()

# 將視覺化輸出
writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

# 開始運算
sess.run(init)
for step in range(201):
    sess.run(train, feed_dict = {x_feeds: x_data, y_feeds: y_data})
    #if step % 20 == 0:
        #print(sess.run(loss, feed_dict = {x_feeds: x_data, y_feeds: y_data}))

sess.close()
```

![day280201](https://storage.googleapis.com/2017_ithome_ironman/day280201.png)

回到終端機可以看到視覺化檔案已經生成，接著利用 `cd` 指令切換到 `"TensorBoard/` 的上一層，執行這段指令：

```
$ tensorboard --logdir='TensorBoard/'
```

等我們看到系統回覆之後，就可以打開瀏覽器，在網址列輸入：localhost:6006 ，就可以在 **Graphs** 頁籤下看到神經網絡圖。

![day2803](https://storage.googleapis.com/2017_ithome_ironman/day2803.png)

## 視覺化（2）

我們除了可以使用 `with tf.name_scope():` 為每個運算元命名，我們還可以在使用 `tf.Variable()` 或者 `tf.placeholder()` 建立變數或輸入資料時，利用 `name = ` 參數進行命名。

```python
import tensorflow as tf
import numpy as np

# 定義一個添加層的函數
def add_layer(inputs, input_tensors, output_tensors, activation_function = None):
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_normal([input_tensors, output_tensors]), name = 'W')
        with tf.name_scope('Biases'):
            b = tf.Variable(tf.zeros([1, output_tensors]), name = 'b')
        with tf.name_scope('Formula'):
            formula = tf.add(tf.matmul(inputs, W), b)
        if activation_function is None:
            outputs = formula
        else:
            outputs = activation_function(formula)
        return outputs

# 準備資料
x_data = np.random.rand(100)
x_data = x_data.reshape(len(x_data), 1)
y_data = x_data * 0.1 + 0.3

# 建立 Feeds
with tf.name_scope('Inputs'):
    x_feeds = tf.placeholder(tf.float32, shape = [None, 1], name = 'x_inputs')
    y_feeds = tf.placeholder(tf.float32, shape = [None, 1], name = 'y_inputs')

# 添加 1 個隱藏層
hidden_layer = add_layer(x_feeds, input_tensors = 1, output_tensors = 10, activation_function = None)

# 添加 1 個輸出層
output_layer = add_layer(hidden_layer, input_tensors = 10, output_tensors = 1, activation_function = None)

# 定義 `loss` 與要使用的 Optimizer
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(y_feeds - output_layer))
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss)

# 初始化 Graph
init = tf.global_variables_initializer()
sess = tf.Session()

# 將視覺化輸出
writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

# 開始運算
sess.run(init)
for step in range(201):
    sess.run(train, feed_dict = {x_feeds: x_data, y_feeds: y_data})
    #if step % 20 == 0:
        #print(sess.run(loss, feed_dict = {x_feeds: x_data, y_feeds: y_data}))

sess.close()
```

![day2804](https://storage.googleapis.com/2017_ithome_ironman/day2804.png)

## 視覺化（3）

我們很快就有了疑問：那麼其他頁籤的功能呢？**TensorBoard** 還能夠將訓練過程視覺化呈現，我們利用 `tf.summary.histogram()` 與 `tf.summary.scalar()` 將訓練過程記錄起來，然後在 **Scalars** 與 **Histograms** 頁籤檢視。

```python
import tensorflow as tf
import numpy as np

# 定義一個添加層的函數
def add_layer(inputs, input_tensors, output_tensors, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_normal([input_tensors, output_tensors]), name = 'W')
            tf.summary.histogram(name = layer_name + '/Weights', values = W)
        with tf.name_scope('Biases'):
            b = tf.Variable(tf.zeros([1, output_tensors]), name = 'b')
            tf.summary.histogram(name = layer_name + '/Biases', values = b)
        with tf.name_scope('Formula'):
            formula = tf.add(tf.matmul(inputs, W), b)
        if activation_function is None:
            outputs = formula
        else:
            outputs = activation_function(formula)
        tf.summary.histogram(name = layer_name + '/Outputs', values = outputs)
        return outputs

# 準備資料
x_data = np.random.rand(100)
x_data = x_data.reshape(len(x_data), 1)
y_data = x_data * 0.1 + 0.3

# 建立 Feeds
with tf.name_scope('Inputs'):
    x_feeds = tf.placeholder(tf.float32, shape = [None, 1], name = 'x_inputs')
    y_feeds = tf.placeholder(tf.float32, shape = [None, 1], name = 'y_inputs')

# 添加 1 個隱藏層
hidden_layer = add_layer(x_feeds, input_tensors = 1, output_tensors = 10, n_layer = 1, activation_function = None)

# 添加 1 個輸出層
output_layer = add_layer(hidden_layer, input_tensors = 10, output_tensors = 1, n_layer = 2, activation_function = None)

# 定義 `loss` 與要使用的 Optimizer
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(y_feeds - output_layer))
    tf.summary.scalar('loss', loss)
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss)

# 初始化 Graph
init = tf.global_variables_initializer()
sess = tf.Session()

# 將視覺化輸出
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

# 開始運算
sess.run(init)
for step in range(400):
    sess.run(train, feed_dict = {x_feeds: x_data, y_feeds: y_data})
    if step % 20 == 0:
        result = sess.run(merged, feed_dict={x_feeds: x_data, y_feeds: y_data})
        writer.add_summary(result, step)

sess.close()
```

![day2805](https://storage.googleapis.com/2017_ithome_ironman/day2805.png)

![day2806](https://storage.googleapis.com/2017_ithome_ironman/day2806.png)

## 小結

第二十八天我們繼續練習 Python 的神經網絡套件 **TensorFlow**，延續昨天的練習程式，使用 **TensorBoard** 來進行神經網絡模型的視覺化。

## 參考連結

- [Tensorflow Python API | TensorFlow](https://www.tensorflow.org/api_docs/python/)
- [TensorBoard: Visualizing Learning | TensorFlow](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/)
- [MorvanZhou@GitHub](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT)