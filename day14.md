# [R 語言使用者的 Python 學習筆記 - 第 14 天] 常用屬性或方法（3）

---

除了 Python 基本的資料結構（list，tuple 與 dictionary）以及昨天學習筆記提到的 ndarray，還記得我們在 [[R 語言使用者的 Python 學習筆記 - 第 05 天] 資料結構（3）](http://ithelp.ithome.com.tw/articles/10185182)提到，為了讓 Python 也能夠使用類似 R 語言的 data frame 資料結構而使用了 `pandas` 套件的 data frame 嗎？我們勢必也要瞭解她常見的屬性或方法。

## Pandas 與 data frame 的常用屬性或方法

### 建立 data frame

使用 `pandas` 套件的 `DataFrame()` 方法將一個 **dictionary** 的資料結構轉換成 data frame。

```python
import pandas as pd

# 截至 2016-12-14 上午 11 時第 8 屆 iT 邦幫忙各組的鐵人分別是 59、9、19、14、6 與 77 人
groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

ironmen_df = pd.DataFrame(ironmen_dict)
ironmen_df
```

![day1401](https://storage.googleapis.com/2017_ithome_ironman/day1401.png)

眼尖的你發現到我們在建立 data frame 的時候並沒有去指定索引值（index），然而生成的 data frame 卻自動產生了類似 R 語言的 `row.names`，多麽貼心的設計！

### 瞭解 data frame 的概觀

- `ndim` 屬性
- `shape` 屬性
- `dtypes` 屬性

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

# 使用屬性
print(ironmen_df.ndim)
print("---") # 分隔線
print(ironmen_df.shape)
print("---") # 分隔線
print(ironmen_df.dtypes)
```

![day1402](https://storage.googleapis.com/2017_ithome_ironman/day1402.png)

### 刪除觀測值或欄位

data frame 可以透過 `drop()` 方法來刪除觀測值或欄位，指定參數 `axis = 0` 表示要刪除觀測值（row），指定參數 `axis = 1` 表示要刪除欄位（column）。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

# 刪除觀測值
ironmen_df_no_mw = ironmen_df.drop(0, axis = 0)
print(ironmen_df_no_mw)
print("---") # 分隔線

# 刪除欄位
ironmen_df_no_groups = ironmen_df.drop("groups", axis = 1)
print(ironmen_df_no_groups)
```

![day1403](https://storage.googleapis.com/2017_ithome_ironman/day1403.png)

### 透過 `ix` 方法篩選 data frame

我們可以透過 `ix` 方法（利用索引值）篩選 data frame。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

# 選擇欄位
print(ironmen_df.ix[:, "groups"])
print("---") # 分隔線

# 選擇觀測值
print(ironmen_df.ix[0])
print("---") # 分隔線

# 同時選擇欄位與觀測值
print(ironmen_df.ix[0, "groups"])
```

![day1404](https://storage.googleapis.com/2017_ithome_ironman/day1404.png)

### 透過布林值篩選 data frame

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

filter = ironmen_df["ironmen"] > 10 # 參賽人數大於 10
ironmen_df[filter] # 篩選 data frame
```

![day1405](https://storage.googleapis.com/2017_ithome_ironman/day1405.png)

### 排序

- `sort_index()` 方法
- `sort_values()` 方法

使用 data frame 的 `sort_index()` 方法可以用索引值排序。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen, columns = ["ironmen"], index = groups)

# 用索引值排序
ironmen_df.sort_index()
```

![day1406](https://storage.googleapis.com/2017_ithome_ironman/day1406.png)

使用 data frame 的 `sort_values()` 方法可以用指定欄位的數值排序。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen, columns = ["ironmen"], index = groups)

# 用數值排序
ironmen_df.sort_values(by = "ironmen")
```

![day1407](https://storage.googleapis.com/2017_ithome_ironman/day1407.png)

### 描述統計

data frame 有 `sum()`、`mean()`、`median()` 與 `describe()` 等統計方法可以使用。

```python
import pandas as pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df.sum()) # 計算總鐵人數
print("---") # 分隔線
print(ironmen_df.mean()) # 計算平均鐵人數
print("---") # 分隔線
print(ironmen_df.median()) # 計算中位數
print("---") # 分隔線
print(ironmen_df.describe()) # 描述統計
```

![day140801](https://storage.googleapis.com/2017_ithome_ironman/day140801.png)

### 相異值個數

透過 `pandas` 的 `value_counts()` 方法可以統計相異值的個數。

```python
import pandas as pd

gender = ["Male", "Male", "Female", "Male", "Male", "Male", "Female", "Male", "Male"]
name = ["蒙其·D·魯夫", "羅羅亞·索隆", "娜美", "騙人布", "文斯莫克·香吉士", "多尼多尼·喬巴", "妮可·羅賓", "佛朗基", "布魯克"]

# 建立 data frame
ironmen_df = pd.DataFrame(gender, columns = ["gender"], index = name)

# 計算男女各有幾個觀測值
pd.value_counts(ironmen_df.gender)
```

![day1409](https://storage.googleapis.com/2017_ithome_ironman/day1409.png)

### 遺失值

#### 判斷遺失值

- `isnull()` 方法
- `notnull()` 方法

```python
import numpy as np
import pandas as pd

groups = ["Modern Web", "DevOps", np.nan, "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, np.nan]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

print(ironmen_df.ix[:, "groups"].isnull()) # 判斷哪些組的組名是遺失值
print("---") # 分隔線
print(ironmen_df.ix[:, "ironmen"].notnull()) # 判斷哪些組的鐵人數不是遺失值
```

![day1410](https://storage.googleapis.com/2017_ithome_ironman/day1410.png)

#### 處理遺失值

- `dropna()` 方法
- `fillna()` 方法

```python
import numpy as np
import pandas as pd

groups = ["Modern Web", "DevOps", np.nan, "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, np.nan]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

ironmen_df_na_dropped = ironmen_df.dropna() # 有遺失值的觀測值都刪除
print(ironmen_df_na_dropped)
print("---") # 分隔線
ironmen_df_na_filled = ironmen_df.fillna(0) # 有遺失值的觀測值填補 0
print(ironmen_df_na_filled)
print("---") # 分隔線
ironmen_df_na_filled = ironmen_df.fillna({"groups": "Cloud", "ironmen": 71}) # 依欄位填補遺失值
print(ironmen_df_na_filled)
```

![day1411](https://storage.googleapis.com/2017_ithome_ironman/day1411.png)

## 小結

第十四天我們討論了 `pandas` 套件與 **data frame** 的屬性或方法，包含建立，篩選與排序等，這些屬性與方法有的隸屬於 `pandas` 套件，有的隸屬於 **data frame** 這個資料結構所建立的物件，對於熟悉物件導向的概念是很好的練習機會。

## 參考連結

- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)
- [pandas: powerful Python data analysis toolkit](http://pandas.pydata.org/pandas-docs/stable/)