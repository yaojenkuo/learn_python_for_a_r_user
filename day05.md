截至 2016-12-05 上午 11 時 第 8 屆 iT 邦幫忙各組的鐵人分別是 46、8、11、11、4 與 56 人，我們想計算參賽鐵人們完賽後各組的總文章數分別是多少。

R 語言我們直接用一個 vector 儲存各組人數後乘以 30，就能得到一個 element-wise 的計算結果，這就是我們要的答案。

```{r}
ironmen <- c(46, 8, 11, 11, 4, 56)
articles <- ironmen * 30
articles
```

![day0501](https://storage.googleapis.com/2017_ithome_ironman/day0501.png)

如果利用 Python 的 list 資料結構，會發現 Python 會輸出 30 次 ironmen，這並不是我們想要的答案。

```{python}
ironmen = [46, 8, 11, 11, 4, 56]
articles = ironmen * 30
print(articles)
```

![day0502](https://storage.googleapis.com/2017_ithome_ironman/day0502.png)

假如我們寫得再謹慎一點，例如將 30 也用 list 包裝起來，這時我們會發現 Python 回傳了一個錯誤訊息。

```{python}
ironmen = [46, 8, 11, 11, 4, 56]
article_multiplier = [30, 30, 30, 30, 30, 30]
articles = ironmen * article_multiplier
print(articles)
```

![day0503](https://storage.googleapis.com/2017_ithome_ironman/day0503.png)

如果我們希望在 Python 輕鬆地使用 element-wise 的運算，我們得仰賴 `numpy` 套件中提供的一種資料結構 **numpy array**，或者採用更精準一點的說法是 **ndarray** 這個資料結構。

## 第一個 numpy 應用

我們來使用 `numpy` 套件中的 **ndarray** 解決先前遭遇到的問題。由於我們的開發環境安裝 [Anaconda](https://www.continuum.io/) ，所以我們不需要再去下載與安裝 `numpy` 套件，我們只需要在程式的上方引用即可（關於本系列文章的 Python 開發環境安裝請參考 [[R 語言使用者的 Python 學習筆記 - 第 01 天] 建立開發環境與計算機應用](http://ithelp.ithome.com.tw/articles/10184561)。）

```{python}
import numpy # 引用套件

ironmen = numpy.array([46, 8, 11, 11, 4, 56]) # 將 list 透過 numpy 的 array 方法進行轉換
print(ironmen) # 看看 ironmen 的外觀
print(type(ironmen)) # 看看 ironmen 的資料結構
articles = ironmen * 30
print(articles)
```

![day0504](https://storage.googleapis.com/2017_ithome_ironman/day0504.png)

R 語言的使用者習慣函數式編程（functional programming），對於 `numpy.array()` 這樣的寫法多少會覺得有些突兀，這時可以與 R 語言中 `package_name::function_name()` 同時指定套件名稱與函數名稱的寫法做對照，瞬間會有恍然大悟的感覺。

為了少打幾個字，我們引用 `numpy` 套件之後依照使用慣例將它縮寫為 `np`。

```{python}
import numpy as np # 引用套件並縮寫為 np

ironmen = np.array([46, 8, 11, 11, 4, 56]) # 將 list 透過 numpy 的 array 方法進行轉換
print(ironmen) # 看看 ironmen 的外觀
print(type(ironmen)) # 看看 ironmen 的資料結構
articles = ironmen * 30
print(articles)
```

![day0505](https://storage.googleapis.com/2017_ithome_ironman/day0505.png)

我們回顧一下 R 語言資料結構中的 vector 與 matrix，然後再研究 Python 的 ndarray。

## R 語言的 vector 與 matrix

### 單一資料類型

R 語言的 vector 與 matrix 都只能容許一種資料類型，如果同時儲存有數值，邏輯值，會被自動轉換為數值，如果同時儲存有數值，邏輯值與文字，會被自動轉換為文字。

```{r}
my_vector <- c(1, TRUE)
class(my_vector) # "numeric"
my_vector <- c(1, TRUE, "one")
class(my_vector) # "character"
```

![day0506](https://storage.googleapis.com/2017_ithome_ironman/day0506.png)

```{r}
my_matrix <- matrix(c(1, 0, TRUE, FALSE), nrow = 2)
my_matrix
my_matrix <- matrix(c(1, "zero", TRUE, FALSE), nrow = 2)
my_matrix
```

![day0507](https://storage.googleapis.com/2017_ithome_ironman/day0507.png)

### Element-wise 運算

R 語言的 vector 與 matrix 完全支持 element-wise 運算。

```{r}
my_vector <- 1:4
my_vector ^ 2

my_matrix <- matrix(my_vector, nrow = 2)
my_matrix ^ 2
```

![day0508](https://storage.googleapis.com/2017_ithome_ironman/day0508.png)

### 選擇元素

R 語言的 vector 與 matrix 都透過中括號 `[]` 或者邏輯值選擇元素。

```{r}
ironmen <- c(46, 8, 11, 11, 4, 56)
ironmen[1] # 選出 Modern Web 組的鐵人數
ironmen > 10 # 哪幾組的鐵人數超過 10 人
ironmen[ironmen > 10] # 超過 10 人的鐵人數
names(ironmen) <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組") # 把 vector 的元素加上名稱
names(ironmen[ironmen > 10]) # 超過 10 人參賽的組別名
```

![day0509](https://storage.googleapis.com/2017_ithome_ironman/day0509.png)

```{r}
ironmen <- c(46, 8, 11, 11, 4, 56)
ironmen_mat <- matrix(ironmen, nrow = 2)
ironmen_mat[1, 1] # 選出 Modern Web 組的鐵人數
ironmen_mat > 10 # 哪幾組的鐵人數超過 10 人
ironmen_mat[ironmen_mat > 10] # 超過 10 人的鐵人數
```

![day0510](https://storage.googleapis.com/2017_ithome_ironman/day0510.png)

### 了解 matrix 規模的函數

R 語言可以透過 `length()` 與 `dim()` 函數來了解 matrix 的規模。

```{r}
ironmen <- c(46, 8, 11, 11, 4, 56)
ironmen_mat <- matrix(ironmen, nrow = 2)
length(ironmen_mat)
dim(ironmen_mat)
```

![day0511](https://storage.googleapis.com/2017_ithome_ironman/day0511.png)

## Python 的 ndarray

### 單一資料類型

NumPy 的 ndarray 只能容許一種資料類型，如果同時儲存有數值，布林值，會被自動轉換為數值，如果同時儲存有數值，布林值與文字，會被自動轉換為文字。

```{python}
import numpy as np

my_np_array = np.array([1, True])
print(my_np_array.dtype) # int64
my_np_array = np.array([1, True, "one"])
print(my_np_array.dtype) # unicode_21
```

![day0512](https://storage.googleapis.com/2017_ithome_ironman/day0512.png)

```{python}
import numpy as np

my_2d_array = np.array([[1, True],
                        [0, False]])
print(my_2d_array)

my_2d_array = np.array([[1, True],
                        ["zero", False]])
print(my_2d_array)
```

![day0513](https://storage.googleapis.com/2017_ithome_ironman/day0513.png)

### Element-wise 運算

NumPy 的 ndarray 完全支持 element-wise 運算。

```{python}
import numpy as np

my_np_array = np.array([1, 2, 3, 4])
print(my_np_array ** 2)

my_2d_array = np.array([[1, 3],
                        [2, 4]])
print(my_2d_array ** 2)
```

![day0514](https://storage.googleapis.com/2017_ithome_ironman/day0514.png)

### 選擇元素

NumPy 的 ndarray 透過中括號 `[]` 或者布林值選擇元素。

```{python}
import numpy as np

ironmen = np.array([46, 8, 11, 11, 4, 56])
print(ironmen[0]) # 選出 Modern Web 組的鐵人數
print(ironmen > 10) # 哪幾組的鐵人數超過 10 人
print(ironmen[ironmen > 10]) # 超過 10 人的鐵人數
```

![day0515](https://storage.googleapis.com/2017_ithome_ironman/day0515.png)

```{python}
import numpy as np

ironmen_2d_array = np.array([[46, 11, 4],
                            [8, 11, 56]])
print(ironmen_2d_array[0, 0]) # 選出 Modern Web 組的鐵人數
print(ironmen_2d_array > 10) # 哪幾組的鐵人數超過 10 人
print(ironmen_2d_array[ironmen_2d_array > 10]) # 超過 10 人的鐵人數
```

![day0516](https://storage.googleapis.com/2017_ithome_ironman/day0516.png)

### 了解 2d array 外觀的函數

NumPy 可以透過 `.size` 與 `.shape` 來了解 2d array 的規模。

```{python}
import numpy as np

ironmen_2d_array = np.array([[46, 11, 4],
                            [8, 11, 56]])
print(ironmen_2d_array.size) # 6
print(ironmen_2d_array.shape) # (2, 3)
```

![day0517](https://storage.googleapis.com/2017_ithome_ironman/day0517.png)

## 小結

第五天我們開始使用 python 的 `numpy` 套件，透過這個套件我們可以使用一種稱為 **ndarray** 的資料結構，透過 **ndarray** 我們可以實現 element-wise 的運算，並且跟 R 語言中的 vector 及 matrix 相互對照。

## 參考連結

- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)
- [NumPy - 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/NumPy)
- [Python For Data Science Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)