# [第 18 天] 資料視覺化 matplotlib

---

在我們昨天的文章 [[第 17 天] 資料角力](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day17.md)有提到，進行資料角力（Data wrangling）的目的多半是為了後續的資料視覺化或者建立機器學習的模型。R 語言使用者的資料視覺化工具有靜態的 **Base plotting system**（R 語言內建的繪圖功能）跟 **ggplot2** 套件，與動態的 **plotly** 套件。而 Python 的視覺化套件有靜態的 **matplotlib** 跟 **seaborn** 套件，與動態的 **bokeh** 套件。

我們今天試著使用看看 **matplotlib** 並且也使用 R 語言的 **Base plotting system** 來畫一些基本的圖形，包括：

- 直方圖（Histogram）
- 散佈圖（Scatter plot）
- 線圖（Line plot）
- 長條圖（Bar plot）
- 盒鬚圖（Box plot）

我們的開發環境是 **Jupyter Notebook**，這個指令可以讓圖形不會在新視窗呈現。

```python
%matplotlib inline
```

## 直方圖（Histogram）

### Python

使用 `matplotlib.pyplot` 的 `hist()` 方法。

```
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
uniform_samples = np.random.uniform(size = 100000) # 生成 100000 組介於 0 與 1 之間均勻分配隨機變數

plt.hist(normal_samples)
plt.show()
plt.hist(uniform_samples)
plt.show()
```

![day1801](https://storage.googleapis.com/2017_ithome_ironman/day1801.png)

![day1802](https://storage.googleapis.com/2017_ithome_ironman/day1802.png)

如果你對於 `numpy` 套件的 `random()` 方法覺得陌生，我推薦你參考 [[第 13 天] 常用屬性或方法（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day13.md)。

### R 語言

使用 `hist()` 函數。

```
normal_samples <- runif(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
uniform_samples <- rnorm(100000) # 生成 100000 組介於 0 與 1 之間均勻分配隨機變數

hist(normal_samples)
hist(uniform_samples)
```

![day1803](https://storage.googleapis.com/2017_ithome_ironman/day1803.png)

![day1804](https://storage.googleapis.com/2017_ithome_ironman/day1804.png)

## 散佈圖（Scatter plot）

### Python

使用 `matplotlib.pyplot` 的 `scatter()` 方法。

```python
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.scatter(speed, dist)
plt.show()
```

![day1805](https://storage.googleapis.com/2017_ithome_ironman/day1805.png)

### R 語言

使用 `plot()` 函數。

```
plot(cars$speed, cars$dist)
```

![day1806](https://storage.googleapis.com/2017_ithome_ironman/day1806.png)

## 線圖（Line plot）

### Python

使用 `matplotlib.pyplot` 的 `plot()` 方法。

```python
%matplotlib inline

import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

plt.plot(speed, dist)
plt.show()
```

![day1807](https://storage.googleapis.com/2017_ithome_ironman/day1807.png)

### R 語言

使用 `plot()` 函數，指定參數 `type = "l"`。

```
plot(cars$speed, cars$dist, type = "l")
```

![day1808](https://storage.googleapis.com/2017_ithome_ironman/day1808.png)

## 長條圖（Bar plot）

### Python

使用 `matplotlib.pyplot` 的 `bar()` 方法。

```python
%matplotlib inline

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]

labels, values = zip(*Counter(cyl).items())
width = 1

plt.bar(indexes, values)
plt.xticks(indexes + width * 0.5, labels)
plt.show()
```

![day1809](https://storage.googleapis.com/2017_ithome_ironman/day1809.png)

### R 語言

使用 `barplot()` 函數。

```
barplot(table(mtcars$cyl))
```

![day1810](https://storage.googleapis.com/2017_ithome_ironman/day1810.png)

## 盒鬚圖（Box plot）

### python

使用 `matplotlib.pyplot` 的 `boxplot()` 方法。

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

plt.boxplot(normal_samples)
plt.show()
```

![day1811](https://storage.googleapis.com/2017_ithome_ironman/day1811.png)

### R 語言

使用 `boxplot()` 函數。

```
normal_samples <- runif(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
boxplot(normal_samples)
```

![day1812](https://storage.googleapis.com/2017_ithome_ironman/day1812.png)

## 輸出圖形

### python

使用圖形物件的 `savefig()` 方法。

```
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數

plt.hist(normal_samples)
plt.savefig(filename = "my_hist.png", format = "png")
```

![day1813](https://storage.googleapis.com/2017_ithome_ironman/day1813.png)

### R 語言

先使用 `png()` 函數建立一個空的 `.png` 圖檔，繪圖後再輸入 `dev.off()`。

```
normal_samples <- runif(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
png("my_hist.png")
hist(normal_samples)
dev.off()
```

![day1814](https://storage.googleapis.com/2017_ithome_ironman/day1814.png)

## 小結

第十八天我們練習使用 Python 的視覺化套件 **matplotlib** 繪製基本的圖形，並且與 R 語言的 **Base plotting system** 相互對照。

## 參考連結

- [pyplot - Matplotlib 1.5.3 documentation](http://matplotlib.org/api/pyplot_api.html)