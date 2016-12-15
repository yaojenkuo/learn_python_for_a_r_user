# [R 語言使用者的 Python 學習筆記 - 第 15 天] 載入資料

---

截至目前為止我們在學習筆記練習的資料結構，不論是 Python 內建的 list，tuple 或 dictionary，還是引用 `numpy` 套件與 `pandas` 套件之後所使用的 ndarray 或 data frame，都是手動創建這些資料，但是在應用資料科學的場景之中，通常不是手動創建資料，而是將資料載入工作環境（R 語言或者 Python），然後再進行後續資料處理與分析。

我們選擇幾種常見的讀入資料格式，分別使用 R 語言與 Python 進行載入。

- csv
- 不同分隔符號的資料
- Excel 試算表
- JSON

## 載入 csv

副檔名為 **.csv** 的資料格式顧名思義是**逗號分隔資料（comma separated values）**，是最常見的表格式資料（tabular data）格式。

![day1501](https://storage.googleapis.com/2017_ithome_ironman/day1501.png)

### R 語言

我們使用 `read.csv()` 函數來載入。

```
url <- "https://storage.googleapis.com/2017_ithome_ironman/data/iris.csv" # 在雲端上儲存了一份 csv 檔案
iris_df <- read.csv(url)
head(iris_df)
```

![day1502](https://storage.googleapis.com/2017_ithome_ironman/day1502.png)

### Python

我們使用 `pandas` 套件的 `read_csv()` 方法來載入。

```python
import pandas as pd

url = "https://storage.googleapis.com/2017_ithome_ironman/data/iris.csv" # 在雲端上儲存了一份 csv 檔案
iris_df = pd.read_csv(url)
iris_df.head()
```

![day1503](https://storage.googleapis.com/2017_ithome_ironman/day1503.png)

## 載入不同分隔符號的資料

除了以**逗號分隔資料**以外，還有不同的方式可以區隔資料欄位，像是以 `tab 鍵（"\t"）` 分隔的資料（tab separated values)，以 `空格（"\s"）` 分隔的資料或者以 `冒號（":"）`分隔的資料，面對這些使用不同的分隔符號（delimeters/separators）的資料，我們可以指定 `sep = ` 這個參數來載入資料。

![day1504](https://storage.googleapis.com/2017_ithome_ironman/day1504.png)

![day1505](https://storage.googleapis.com/2017_ithome_ironman/day1505.png)

### R 語言

我們使用 `read.table()` 函數來載入，並且依據分隔符號指定 `sep = `參數。

#### 以 tab 鍵（"\t"）分隔

```
url <- "https://storage.googleapis.com/2017_ithome_ironman/data/iris.tsv" # 在雲端上儲存了一份 tsv 檔案
iris_tsv_df <- read.table(url, sep = "\t", header = TRUE)
head(iris_tsv_df)
```

![day1506](https://storage.googleapis.com/2017_ithome_ironman/day1506.png)

#### 以冒號（":"）分隔

```
url <- "https://storage.googleapis.com/2017_ithome_ironman/data/iris.txt" # 在雲端上儲存了一份 txt 檔案
iris_colon_sep_df <- read.table(url, sep = ":", header = TRUE)
head(iris_colon_sep_df)
```

![day1507](https://storage.googleapis.com/2017_ithome_ironman/day1507.png)

### Python

我們使用 `pandas` 套件的 `read_table()` 方法來載入，並且依據分隔符號指定 `sep = `參數。

#### 以 tab 鍵（"\t"）分隔

```python
import pandas as pd

url = "https://storage.googleapis.com/2017_ithome_ironman/data/iris.tsv" # 在雲端上儲存了一份 tsv 檔案
iris_tsv_df = pd.read_table(url, sep = "\t")
iris_tsv_df.head()
```

![day1508](https://storage.googleapis.com/2017_ithome_ironman/day1508.png)

#### 以冒號（":"）分隔

```python
import pandas as pd

url = "https://storage.googleapis.com/2017_ithome_ironman/data/iris.txt" # 在雲端上儲存了一份 txt 檔案
iris_colon_sep_df = pd.read_table(url, sep = ":")
iris_colon_sep_df.head()
```

![day1509](https://storage.googleapis.com/2017_ithome_ironman/day1509.png)

## 載入 Excel 試算表

我們以副檔名為 `.xlsx` 的 Excel 試算表檔案為例。

![day1510](https://storage.googleapis.com/2017_ithome_ironman/day1510.png)

### R 語言

我們使用 `readxl` 套件的 `read_excel()` 函數來載入。

```
library(readxl)

file_path <- "~/Downloads/iris.xlsx" # read_excel 暫時不支援 https 先將試算表下載到本機 https://storage.googleapis.com/2017_ithome_ironman/data/iris.xlsx

iris_xlsx_df <- read_excel(file_path)
head(iris_xlsx_df)
```

![day1511](https://storage.googleapis.com/2017_ithome_ironman/day1511.png)

### Python

我們使用 `pandas` 套件的 `read_excel()` 方法來載入。

```python
import pandas as pd

url = "https://storage.googleapis.com/2017_ithome_ironman/data/iris.xlsx" # 在雲端上儲存了一份 Excel 試算表
iris_xlsx_df = pd.read_excel(url)
iris_xlsx_df.head()
```

![day1512](https://storage.googleapis.com/2017_ithome_ironman/day1512.png)

## 載入 JSON

JSON（JavaScript Object Notation）格式的資料是網站資料傳輸以及 NoSQL（Not only SQL）資料庫儲存的主要類型，R 語言與 Python 有相對應的套件可以協助我們把 JSON 資料格式載入後轉換為我們熟悉的 data frame。

![day1513](https://storage.googleapis.com/2017_ithome_ironman/day1513.png)

### R 語言

我們使用 `jsonlite` 套件的 `fromJSON()` 函數來載入。

```
library(jsonlite)

url <- "https://storage.googleapis.com/2017_ithome_ironman/data/iris.json" # 在雲端上儲存了一份 JSON 檔
iris_json_df <- fromJSON(url)
head(iris_json_df)
```

![day1514](https://storage.googleapis.com/2017_ithome_ironman/day1514.png)

### Python

我們使用 `pandas` 套件的 `read_json()` 方法來載入。

```python
import pandas as pd

url = "https://storage.googleapis.com/2017_ithome_ironman/data/iris.json" # 在雲端上儲存了一份 JSON 檔
iris_json_df = pd.read_json(url)
iris_json_df.head()
```

![day1515](https://storage.googleapis.com/2017_ithome_ironman/day1515.png)

## 小結

第十五天我們討論如何將 csv，不同分隔符號的資料，Excel 試算表與 JSON 格式的資料讀入 Python，我們透過 `pandas` 套件的 `read_csv()`、`read_table()`、`read_excel()` 與 `read_json` 等方法可以將不同的資料格式轉換為我們熟悉的 data frame 資料結構，同時我們也跟 R 語言讀入不同資料格式的各種函數作對照。

## 參考連結

- [JSON - 維基百科，自由的百科全書](https://zh.wikipedia.org/wiki/JSON)
- [Getting started with JSON and jsonlite](https://cran.r-project.org/web/packages/jsonlite/vignettes/json-aaquickstart.html)
- [API Reference - pandas 0.19.1 documentation](http://pandas.pydata.org/pandas-docs/stable/api.html)
- <https://cran.r-project.org/web/packages/readxl/readxl.pdf>