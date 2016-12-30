# [第 06 天] 資料結構（3）Data Frame

---

截至 2016-12-06 上午 7 時第 8 屆 iT 邦幫忙各組的鐵人分別是 46、8、12、12、6 與 58 人，我們想要用一個表格來紀錄參賽的組別與鐵人數。

R 語言我們可以用 data frame 的資料結構來呈現這個資訊，使用 `data.frame()` 函數可以將 vectors 結合成資料框。

```
groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
ironmen <- c(46, 8, 12, 12, 6, 58)
ironmen_df <- data.frame(groups, ironmen)
ironmen_df
View(ironmen_df)
```

![day0601](https://storage.googleapis.com/2017_ithome_ironman/day0601.png)

![day0602](https://storage.googleapis.com/2017_ithome_ironman/day0602.png)

如果我們希望在 Python 中也能夠使用 data frame，我們得仰賴 `pandas` 套件。與第五天討論的 `numpy` 套件一樣，由於我們的開發環境是安裝 [Anaconda](https://www.continuum.io/)，所以不需要再去下載與安裝 `pandas` 套件，只需要在程式的上方引用即可（關於本系列文章的 Python 開發環境安裝請參考 [[第 01 天] 建立開發環境與計算機應用](http://ithelp.ithome.com.tw/articles/10184561)。）

## 第一個 pandas 應用

我們引用 `pandas` 套件之後依照使用慣例將它縮寫為 `pd`，最基本建立 data frame 的方式是利用 `pandas` 套件的 `DataFrame()` 方法將一個 dictionary 的資料結構轉換為 data frame（如果你對於 dictionary 資料結構感到陌生，我推薦你閱讀[[[第 04 天] 資料結構 List，Tuple 與 Dictionary](http://ithelp.ithome.com.tw/articles/10185010)。）

```python
import pandas as pd # 引用套件並縮寫為 pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df) # 看看資料框的外觀
print(type(ironmen_df)) # pandas.core.frame.DataFrame
```

![day0603](https://storage.googleapis.com/2017_ithome_ironman/day0603.png)

R 語言的使用者如果仍舊對於 `pd.DataFrame()` 的寫法覺得不習慣，這時可以與 R 語言中 `package_name::function_name()` 同時指定套件名稱與函數名稱的寫法做對照，瞬間會有恍然大悟的感覺。

我們回顧一下 R 語言資料結構中的 data frame，然後再研究 `pandas` 套件的 data frame。

## R 語言的 data frame

### 可以由 matrix 轉換

如果 data frame 的變數類型相同，亦可以從 matrix 轉換。

```
my_mat <- matrix(1:4, nrow = 2, dimnames = list(NULL, c("col1", "col2")))
my_df <- data.frame(my_mat)
my_df
```

![day0604](https://storage.googleapis.com/2017_ithome_ironman/day0604.png)

### 包含多種資料類型

跟 list 的特性相仿，不會像 vector 與 matrix 僅限制容納單一資料類型。

```
groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
ironmen <- c(46, 8, 12, 12, 6, 58)
ironmen_df <- data.frame(groups, ironmen)

sapply(ironmen_df, FUN = class) # 回傳每一個欄位的 class
```

![day0605](https://storage.googleapis.com/2017_ithome_ironman/day0605.png)

### 選擇元素

R 語言透過使用中括號 `[ , ]` 或者 `$` 可以很靈活地從 data frame 中選擇想要的元素（值，列或欄。）

```
groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
ironmen <- c(46, 8, 12, 12, 6, 58)
ironmen_df <- data.frame(groups, ironmen)

ironmen_df[1, 2] # 第一列第二欄：Modern Web 組的鐵人數
ironmen_df[1, ] # 第一列：Modern Web 組的組名與鐵人數
ironmen_df[, 2] # 第二欄：各組的鐵人數
ironmen_df[, "ironmen"] # 各組的鐵人數
ironmen_df$ironmen # 各組的鐵人數
```

![day0606](https://storage.googleapis.com/2017_ithome_ironman/day0606.png)

### 可以使用邏輯值篩選

R 語言可以透過邏輯值來針對 data frame 進行觀測值的篩選。

```
roups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
ironmen <- c(46, 8, 12, 12, 6, 58)
ironmen_df <- data.frame(groups, ironmen)

ironmen_df[ironmen_df$ironmen > 10, ] # 選出鐵人數超過 10 的 data frame
```

![day0607](https://storage.googleapis.com/2017_ithome_ironman/day0607.png)

### 了解 data frame 概觀的函數

R 語言可以透過一些函數來了解 data frame 的概觀。

```
dim(ironmen_df) # 回傳列數與欄數
str(ironmen_df) # 回傳結構
summary(ironmen_df) # 回傳描述性統計
head(ironmen_df, n = 3) # 回傳前三筆觀測值
tail(ironmen_df, n = 3) # 回傳後三筆觀測值
names(ironmen_df) # 回傳欄位名稱
```

![day0608](https://storage.googleapis.com/2017_ithome_ironman/day0608.png)

## Pandas 的 data frame

### 可以由 NumPy 的 2d array 轉換

如果 data frame 的變數類型相同，亦可以從 NumPy 的 2d array 轉換。

```python
import numpy as np
import pandas as pd

my_2d_array = np.array([[1, 3],
                        [2, 4]
                       ])

my_df = pd.DataFrame(my_2d_array, columns = ["col1", "col2"])
print(my_df)
```

![day0609](https://storage.googleapis.com/2017_ithome_ironman/day0609.png)

### 包含多種資料類型

跟 list 的特性相仿，不會像 ndarray 僅限制容納單一資料類型。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)
print(ironmen_df.dtypes) # 欄位的變數類型
```

![day0610](https://storage.googleapis.com/2017_ithome_ironman/day0610.png)

### 選擇元素

Pandas 透過使用中括號 `[]` 與 `.iloc` 可以很靈活地從 data frame 中選擇想要的元素。要注意的是 Python 在指定 `0:1` 時不包含 `1`，在指定 `0:2` 時不包含 `2`，這一點是跟 R 語言有很大的不同之處。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df.iloc[0:1, 1]) # 第一列第二欄：Modern Web 組的鐵人數
print("---")
print(ironmen_df.iloc[0:1,:]) # 第一列：Modern Web 組的組名與鐵人數
print("---")
print(ironmen_df.iloc[:,1]) # 第二欄：各組的鐵人數
print("---")
print(ironmen_df["ironmen"]) # 各組的鐵人數
print("---")
print(ironmen_df.ironmen) # 各組的鐵人數
```

![day0611](https://storage.googleapis.com/2017_ithome_ironman/day0611.png)

最後兩行我們用了簡便的選擇語法，但是在正式環境選擇元素時仍然推薦使用最合適的 pandas 方法 `.iloc` 與 `.loc` 等。

> While standard Python / Numpy expressions for selecting and setting are intuitive and come in handy for interactive work, for production code, we recommend the optimized pandas data access methods, `.at`, `.iat`, `.loc`, `.iloc` and `.ix`. 
> Quoted from: [10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html#selection)

### 可以使用布林值篩選

Pandas 可以透過布林值來針對 data frame 進行觀測值的篩選。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df[ironmen_df.loc[:,"ironmen"] > 10]) # 選出鐵人數超過 10 的 data frame
```

![day0612](https://storage.googleapis.com/2017_ithome_ironman/day0612.png)

### 了解 data frame 概觀

Pandas 的 data frame 資料結構有一些方法或屬性可以幫助我們了解概觀。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [46, 8, 12, 12, 6, 58]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df.shape) # 回傳列數與欄數
print("---")
print(ironmen_df.describe()) # 回傳描述性統計
print("---")
print(ironmen_df.head(3)) # 回傳前三筆觀測值
print("---")
print(ironmen_df.tail(3)) # 回傳後三筆觀測值
print("---")
print(ironmen_df.columns) # 回傳欄位名稱
print("---")
print(ironmen_df.index) # 回傳 index
```

![day0613](https://storage.googleapis.com/2017_ithome_ironman/day0613.png)

## R 語言的 factor 資料結構與 pandas 的 category 資料結構

### Nominal

截至上一個段落，我們已經對照了四種 R 語言的基本資料結構，還剩下最後一個是 R 語言的 factor，這在 pandas 中可以對照為 category。

```
groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
groups_factor <- factor(groups) # 轉換為 factor
groups_factor
```

![day0614](https://storage.googleapis.com/2017_ithome_ironman/day0614.png)

我們利用 `pandas` 套件的 `Categorical()` 方法轉換 list 為 pandas 的 category 資料結構。

```python
import pandas as pd

groups_categorical = pd.Categorical(["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"])
print(groups_categorical)
print("---")
print(type(groups_categorical))
```

![day0615](https://storage.googleapis.com/2017_ithome_ironman/day0615.png)

### Ordinal

R 語言使用 `ordered = TRUE` 與指定 `levels = ` 參數加入 ordinal 的特性。

```
temperature <- c("cold", "warm", "hot")
temperature_factor <- factor(temperature, ordered = TRUE, levels = c("cold", "warm", "hot"))
temperature_factor
```

![day0616](https://storage.googleapis.com/2017_ithome_ironman/day0616.png)

Pandas 使用 `ordered = True` 與指定 `categories = ` 參數加入 ordinal 的特性。

```python
import pandas as pd

temperature_list = ["cold", "warm", "hot"]
temperature_categorical = pd.Categorical(temperature_list, categories = ["cold", "warm", "hot"], ordered = True)
temperature = pd.Series(temperature_categorical)
print(temperature)
```

![day0617](https://storage.googleapis.com/2017_ithome_ironman/day0617.png)

## 小結

第六天我們開始使用 Python 的 `pandas` 套件，透過這個套件我們可以在 Python 開發環境中使用 data frame 與 category 的資料結構，並且跟 R 語言中的 data frame 及 factor 相互對照。

## 參考連結

- [10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
- [Categorical Data](https://pandas-docs.github.io/pandas-docs-travis/categorical.html)
- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)