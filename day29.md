# [第 29 天] 深度學習（3）MNIST 手寫數字辨識

---

我們今天繼續練習神經網絡的套件 **TensorFlow**，在學習過程中，不論是視覺化或者機器學習的主題，我們使用了幾個常見的玩具資料（Toy datasets），像是 **iris** 鳶尾花資料或者 **cars** 車速與煞車距離資料，這些玩具資料簡潔且易懂，可以讓我們很快速地入門，例如實作迴歸時使用 **cars**，實作分類與分群時使用 **iris**。同樣在深度學習領域也有一個經典的 **MNIST** 手寫數字辨識資料，供初學者實作圖片分類器。

## 讀入 MNIST

如同在 **scikit-learn** 套件中讀入 **iris** 一般，在 **TensorFlow** 套件中讀入 **MNIST** 同樣是很容易的，不論是訓練資料或者測試資料，都有分 `images` 與 `labels` 屬性，我們簡單跟 **scikit-learn** 套件做個對照：

套件|自變數 X|目標變數 y
---|---|---
`sklearn`|data|target
`tensorflow`|images|labels

```python
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 檢視結構
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")

# 檢視一個觀測值
#print(x_train[1, :])
print(np.argmax(y_train[1, :])) # 第一張訓練圖片的真實答案
```

![day2901](https://storage.googleapis.com/2017_ithome_ironman/day2901.png)

**MNIST** 的圖片是 28 像素 x 28 像素，每一張圖片就可以用 28 x 28 = 784 個數字來紀錄，因此 `print(x_train.shape)` 的輸出告訴我們有 55,000 張訓練圖片，每張圖片都有 784 個數字；而 `print(y_train.shape)` 的輸出告訴我們的是這 55,000 張訓練圖片的真實答案，`print(np.argmax(y_train[1, :]))` 的輸出告訴我們第一張訓練圖片的真實答案為 **3**。

我們也可以使用 `matplotlib.pyplot` 把第一張訓練圖片印出來看看。

```python
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images

# 印出來看看
first_train_img = np.reshape(x_train[1, :], (28, 28))
plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
plt.show()
```

![day2906](https://storage.googleapis.com/2017_ithome_ironman/day2906.png)

## Softmax 函數

我們需要透過 Softmax 函數將分類器輸出的分數（Evidence）轉換為機率（Probability），然後依據機率作為預測結果的輸出，可想而知深度學習模型的輸出層會是一個 Softmax 函數。

![day2902](https://storage.googleapis.com/2017_ithome_ironman/day2902.png)

## Cross-entropy

不同於我們先前使用 **Mean Squared Error** 定義 **Loss**，在這個深度學習模型中我們改用 **Cross-entropy** 來定義 **Loss**。

> One very common, very nice function to determine the loss of a model is called "cross-entropy." Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning.
> [MNIST For ML Beginners | TensorFlow](https://www.tensorflow.org/tutorials/mnist/beginners/)

## TensorFlow 實作

我們建立一個可以利用 **TensorBoard** 檢視的深度學習模型，實作手寫數字辨識的分類器。

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
learning_rate = 0.5
training_steps = 1000
batch_size = 100
logs_path = 'TensorBoard/'
n_features = x_train.shape[1]
n_labels = y_train.shape[1]

# 建立 Feeds
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32, [None, n_features], name = 'Input_Data')
with tf.name_scope('Labels'):
    y = tf.placeholder(tf.float32, [None, n_labels], name = 'Label_Data')

# 建立 Variables
with tf.name_scope('ModelParameters'):
    W = tf.Variable(tf.zeros([n_features, n_labels]), name = 'Weights')
    b = tf.Variable(tf.zeros([n_labels]), name = 'Bias')

# 開始建構深度學習模型
with tf.name_scope('Model'):
    # Softmax
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)
with tf.name_scope('CrossEntropy'):
    # Cross-entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices = 1))
    tf.summary.scalar("Loss", loss)
with tf.name_scope('GradientDescent'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', acc)

# 初始化
init = tf.global_variables_initializer()

# 開始執行運算
sess = tf.Session()
sess.run(init)

# 將視覺化輸出
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

# 訓練
for step in range(training_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
    if step % 50 == 0:
        print(sess.run(loss, feed_dict = {x: batch_xs, y: batch_ys}))
        summary = sess.run(merged, feed_dict = {x: batch_xs, y: batch_ys})
        writer.add_summary(summary, step)

print("---")
# 準確率
print("Accuracy: ", sess.run(acc, feed_dict={x: x_test, y: y_test}))

sess.close()
```

![day2903](https://storage.googleapis.com/2017_ithome_ironman/day2903.png)

![day2904](https://storage.googleapis.com/2017_ithome_ironman/day2904.png)

![day2905](https://storage.googleapis.com/2017_ithome_ironman/day2905.png)

如果你對於如何產生 **TensorBoard** 視覺化有興趣，我推薦你參考[昨天](http://ithelp.ithome.com.tw/articles/10187814)的學習筆記。我們的模型準確率有 **92%** 左右，感覺還不錯，但是官方文件卻跟我們說這很糟：

> Getting 92% accuracy on MNIST is bad. It's almost embarrassingly bad.
> [Deep MNIST for Experts | TensorFlow](https://www.tensorflow.org/tutorials/mnist/pros/)

我們明天來試著依照官方文件的教學建立一個卷積神經網絡（Convolutional Neural Network，CNN）提升 **MNIST** 資料的數字辨識準確率。

## 小結

第二十九天我們繼續練習 Python 的深度學習套件 **TensorFlow**，針對 **MNIST** 資料建立了一個神經網絡模型，達到 **92%** 的準確率，同時我們也用了 **TensorBoard** 來視覺化。

## 參考連結

- [MNIST For ML Beginners | TensorFlow](https://www.tensorflow.org/tutorials/mnist/beginners/)
- [Deep MNIST for Experts | TensorFlow](https://www.tensorflow.org/tutorials/mnist/pros/)
- [aymericdamien@GitHub](https://github.com/aymericdamien/TensorFlow-Examples)
- [Tensorflow Day3 : 熟悉 MNIST 手寫數字辨識資料集](http://ithelp.ithome.com.tw/articles/10186473)