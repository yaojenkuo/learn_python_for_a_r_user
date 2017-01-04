# [第 13 天] 常用屬性或方法（2）ndarray

---

我們在昨天的學習筆記討論了 Python 基本變數類型與資料結構可以應用的屬性或方法，除了基本的資料結構以外，你是否還記得 Python 可以透過引入 `numpy` 套件之後使用 **ndarray** 資料結構呢？當時我們在 [[第 05 天] 資料結構（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day05.md) 提到，為了解決 Python 的 list 資料結構無法進行 element-wise 的運算，因此使用了 `numpy` 套件的 **ndarray**，我們勢必要瞭解她常見的屬性或方法。

## numpy 與 ndarray 的常用屬性或方法

### 瞭解 ndarray 的概觀

- `ndim` 屬性
- `shape` 屬性
- `dtype` 屬性

```python
import numpy as np

# 截至 2016-12-06 上午 7 時第 8 屆 iT 邦幫忙各組的鐵人分別是 56、8、19、14、6 與 71 人
ironmen = [56, 8, 19, 14, 6, 71]
ironmen_array = np.array(ironmen)

print(ironmen_array.ndim) # number of dimensions
print(ironmen_array.shape) # m*n
print(ironmen_array.dtype) # 資料類型
print("\n") # 空一行

# 2d array
ironmen_2d = [range(1, 7), [56, 8, 19, 14, 6, 71]]
ironmen_2d_array = np.array(ironmen_2d)
print(ironmen_2d_array.ndim) # number of dimensions
print(ironmen_2d_array.shape) # m*n
print(ironmen_2d_array.dtype) # 資料類型
```

![day1301](https://storage.googleapis.com/2017_ithome_ironman/day1301.png)

### 建立 ndarray

`numpy` 套件除了 `array()` 方法可以將 list 轉換成 ndarray，還有其他的方法可以建立 ndarray。

- `zeros()` 方法
- `empty()` 方法
- `arange()` 方法

```python
import numpy as np

print(np.zeros(6)) # 六個元素均為零的 1d array
print("------") # 分隔線
print(np.zeros((2, 6))) # 十二個元素均為零的 2d array
print("------") # 分隔線
print(np.empty((2, 6, 2))) # 二十四個元素均為未初始化的值
print("------") # 分隔線
print(np.arange(11)) # 十一個元素為 0 到 10 的 1d array
```

![day1302](https://storage.googleapis.com/2017_ithome_ironman/day1302.png)

### 轉換變數類型

ndarray 的 `astype()` 方法可以轉換變數類型。

```python
import numpy as np

ironmen = ["56", "8", "19", "14", "6", "71"]
ironmen_str_array = np.array(ironmen)
print(ironmen_str_array.dtype)
print("---") # 分隔線

# 轉換為 int64
ironmen_int_array = ironmen_str_array.astype(np.int64)
print(ironmen_int_array.dtype)
```

![day1303](https://storage.googleapis.com/2017_ithome_ironman/day1303.png)

### 用索引值進行篩選

利用 `[]` 搭配索引值篩選 ndarray，這點與 R 語言作法相同。

```python
import numpy as np

my_array = np.arange(10)
print(my_array[0])
print(my_array[0:5])
print("---") # 分隔線

my_2d_array = np.array([np.arange(0, 5), np.arange(5, 10)])
print(my_2d_array)
print("---") # 分隔線
print(my_2d_array[1, :]) # 第二列
print(my_2d_array[:, 1]) # 第二欄
print(my_2d_array[1, 1]) # 第二列第二欄的元素
```

![day1304](https://storage.googleapis.com/2017_ithome_ironman/day1304.png)

### 用布林值進行篩選

利用布林值（bool）篩選 ndarray，，這點與 R 語言作法相同。

```python
import numpy as np

ironmen = [56, 8, 19, 14, 6, 71]
groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen_array = np.array(ironmen)
groups_array = np.array(groups)

# 用人數去篩選組別
print(ironmen_array >= 10) # 布林值陣列
print(groups_array[ironmen_array >= 10]) # 鐵人數大於 10 的組別

# 用組別去篩選人數
print(groups_array != "自我挑戰組") # 布林值陣列
print(ironmen_array[groups_array != "自我挑戰組"]) # 除了自我挑戰組以外的鐵人數
```

![day1305](https://storage.googleapis.com/2017_ithome_ironman/day1305.png)

### 2d array 轉置

使用 `T` 屬性。

```python
import numpy as np

# 建立一個 2d array
my_1d_array = np.arange(10)
my_2d_array = my_1d_array.reshape((2, 5))
print(my_2d_array)
print("---") # 分隔線
print(my_2d_array.T)
```

![day1306](https://storage.googleapis.com/2017_ithome_ironman/day1306.png)

### numpy 的 where 方法

透過 `numpy` 的 `where()` 方法在 ndarray 中進行流程控制。

```python
import numpy as np

ironmen_array = np.array([56, 8, 19, 14, 6, np.nan])
np.where(np.isnan(ironmen_array), 71, ironmen_array)
```

![day1307](https://storage.googleapis.com/2017_ithome_ironman/day1307.png)

### 排序

透過 `sort()` 方法。

```python
import numpy as np

ironmen_array = np.array([56, 8, 19, 14, 6, 71])
print(ironmen_array)
ironmen_array.sort()
print(ironmen_array)
```

![day1308](https://storage.googleapis.com/2017_ithome_ironman/day1308.png)

### 隨機變數

透過 `numpy` 的 `random()` 方法可以生成隨機變數。

```python
import numpy as np

normal_samples = np.random.normal(size = 10) # 生成 10 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
uniform_samples = np.random.uniform(size = 10) # 生成 10 組介於 0 與 1 之間均勻分配隨機變數

print(normal_samples)
print("---") # 分隔線
print(uniform_samples)
```

![day1309](https://storage.googleapis.com/2017_ithome_ironman/day1309.png)

## 小結

第十三天我們討論了 `numpy` 套件與 **ndarray** 的屬性或方法，包含建立，變數類型轉換，篩選與排序等，這些屬性與方法有的隸屬於 `numpy` 套件，有的隸屬於 **ndarray** 這個資料結構所建立的物件，對於熟悉物件導向的概念是很好的練習機會。

## 參考連結

- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)
- [Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)