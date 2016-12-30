# [第 19 天] 資料視覺化（2）Seaborn

---

使用 **matplotlib** 建立一個圖表的概念是組裝它提供的基礎元件，像是圖表類型、圖例或者標籤等元件。 **Seaborn** 套件是以 **matplotlib** 為基礎建構的高階繪圖套件，讓使用者更加輕鬆地建立圖表，我們可以將它視為是 **matplotlib** 的補強，如果你對 **matplotlib** 套件有點陌生，我推薦你閱讀 [[第 18 天] 資料視覺化 matplotlib](http://ithelp.ithome.com.tw/articles/10186484)。

> Seaborn is a library for making attractive and informative statistical graphics in Python. It is built on top of matplotlib and tightly integrated with the PyData stack, including support for numpy and pandas data structures and statistical routines from scipy and statsmodels.
> [Seaborn: statistical data visualization](http://seaborn.pydata.org/index.html)

我們今天試著使用看看 **Seaborn** 套件並且也使用 R 語言的 **ggplot2** 套件來畫一些基本的圖形，包括：

- 直方圖（Histogram）
- 散佈圖（Scatter plot）
- 線圖（Line plot）
- 長條圖（Bar plot）
- 盒鬚圖（Box plot）

**Seaborn** 套件在我們的開發環境沒有安裝，但我們可以透過 `conda` 指令在終端機安裝。

```
$ conda install -c anaconda seaborn=0.7.1
```

我們的開發環境是 Jupyter Notebook，這個指令可以讓圖形不會在新視窗呈現。

```
%matplotlib inline
```

## 直方圖（Histogram）

### Python

使用 `seaborn` 套件的 `distplot()` 方法。

```python
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
sns.distplot(normal_samples)
```

![day1901](https://storage.googleapis.com/2017_ithome_ironman/day1901.png)

預設會附上 **kernel density estimate（KDE）**曲線。

### R 語言

使用 `ggplot2` 套件的 `geom_histogram()` 函數指定為直方圖。

```
library(ggplot2)

normal_samples <- rnorm(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples_df <- data.frame(normal_samples)
ggplot(normal_samples_df, aes(x = normal_samples)) + geom_histogram(aes(y = ..density..)) + geom_density()
```

![day1902](https://storage.googleapis.com/2017_ithome_ironman/day1902.png)

## 散佈圖（Scatter plot）

### Python

使用 `seaborn` 套件的 `joinplot()` 方法。

```python
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.jointplot(x = "speed", y = "dist", data = cars_df)
```

![day1903](https://storage.googleapis.com/2017_ithome_ironman/day1903.png)

預設會附上 X 軸變數與 Y 軸變數的直方圖。

### R 語言

使用 `ggplot2` 套件的 `geom_point()` 函數指定為散佈圖，再使用 `ggExtra` 套件的 `ggMarginal()` 函數加上 X 軸變數與 Y 軸變數的直方圖。

```
library(ggplot2)
library(ggExtra)

scatter_plot <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
ggMarginal(scatter_plot, type = "histogram")
```

![day1904](https://storage.googleapis.com/2017_ithome_ironman/day1904.png)

## 線圖（Line plot）

### Python

使用 `seaborn` 套件的 `factorplot()` 方法。

```
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.factorplot(data = cars_df, x="speed", y="dist", ci = None)
```

![day1905](https://storage.googleapis.com/2017_ithome_ironman/day1905.png)

### R 語言

使用 `ggplot2` 套件的 `geom_line()` 函數指定為線圖。

```
library(ggplot2)

ggplot(cars, aes(x = speed, y = dist)) + geom_line()
```

![day1906](https://storage.googleapis.com/2017_ithome_ironman/day1906.png)

## 長條圖（Bar plot）

### Python

使用 `seaborn` 套件的 `countplot()` 方法。

```python
%matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]
cyl_df = pd.DataFrame({"cyl": cyl})

sns.countplot(x = "cyl", data=cyl_df)
```

![day1907](https://storage.googleapis.com/2017_ithome_ironman/day1907.png)

### R 語言

使用 `ggplot2` 套件的 `geom_bar()` 函數指定為長條圖。

```
library(ggplot2)

ggplot(mtcars, aes(x = cyl)) + geom_bar()
```

![day1908](https://storage.googleapis.com/2017_ithome_ironman/day1908.png)

## 盒鬚圖（Box plot）

### Python

使用 `seaborn` 套件的 `boxplot()` 方法。

```python
%matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
sns.boxplot(normal_samples)
```

![day1909](https://storage.googleapis.com/2017_ithome_ironman/day1909.png)

### R 語言

使用 `ggplot2` 套件的 `geom_boxplot()` 函數指定為盒鬚圖。

```
library(ggplot2)

normal_samples <- rnorm(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples_df <- data.frame(normal_samples)
ggplot(normal_samples_df, aes(y = normal_samples, x = 1)) + geom_boxplot() + coord_flip()
```

![day1910](https://storage.googleapis.com/2017_ithome_ironman/day1910.png)

## 小結

第十九天我們練習使用 Python 的視覺化套件 **Seaborn** 繪製基本的圖形，並且與 R 語言的 **ggplot2** 相互對照。

## 參考連結

- [Seaborn: statistical data visualization](http://seaborn.pydata.org/index.html)
- [ggplot2 0.9.3.1](http://docs.ggplot2.org/0.9.3.1/index.html)
- [Seaborn :: Anaconda Cloud](https://anaconda.org/anaconda/seaborn)