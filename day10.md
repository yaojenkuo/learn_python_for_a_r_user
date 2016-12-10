# [R 語言使用者的 Python 學習筆記 - 第 10 天] 物件導向

---

> R, at its heart, is a functional programming language.
> [Hadley Wickham](http://hadley.nz/)

R 語言本質上是一個函數型程式語言，而 Python 是一個物件導向程式語言，這也是我認為 R 語言使用者在學習 Python 的時候會感到較為困惑的部分，尤其在面對類別（Class），屬性（Attribute）或者方法（Method）等陌生辭彙的時候，讓我們先用之前做過的練習切入。

截至 2016-12-10 下午 1 時第 8 屆 iT 邦幫忙各組的鐵人分別是 51、8、18、14、6 與 64 人，我們用一個 data frame 來紀錄參賽的組別與鐵人數，如果你對 data frame 這個資料結構有疑惑，我推薦你閱讀 [[R 語言使用者的 Python 學習筆記 - 第 06 天] 資料結構（3）](http://ithelp.ithome.com.tw/articles/10185182)。

```{r}
groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
ironmen <- c(51, 8, 18, 14, 6, 64)

# 跟 Python 程式比較這兩行
ironmen_df <- data.frame(groups, ironmen)
head(ironmen_df, n = 3)
```

![day1001](https://storage.googleapis.com/2017_ithome_ironman/day1001.png)

```{python}
import pandas as pd # 引用套件並縮寫為 pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [51, 8, 18, 14, 6, 64]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

# 跟 R 語言程式比較這兩行
ironmen_df = pd.DataFrame(ironmen_dict)
ironmen_df.head(n = 3)
```

![day1002](https://storage.googleapis.com/2017_ithome_ironman/day1002.png)

觀察程式的最後兩行，我們可以稍微地感受到函數型程式語言與物件導向程式語言的不同：

- 同樣為建立 data frame，R 語言使用 `data.frame()` 函數；Python 使用 `pd` 的 `DataFrame()` 方法
- 同樣為顯示 data frame 的前三列觀測值，R 語言使用 `head()` 函數；Python 使用 `ironmen_df` 的 `head()` 方法

今天我們先討論 R 語言的物件導向。

## R 語言的物件導向

R 語言的物件導向有三大類別（Class）：

- S3 類別
- S4 類別
- RC（Reference class）

除此之外，還有一個 Base types 類別是只有核心開發團隊才可以新增類別的物件導向類別，所以沒有把它列在上面的清單之中。

### Base types 類別

利用 `typeof()` 與 `is.primitive()` 函數可以驗證什麼物件是屬於 Base types 類別。

```{r}
# 自訂函數不屬於 base types
my_square <- function(input_num) {
    return(input_num^2)
}
typeof(my_square)
is.primitive(my_square)

# sum() 函數屬於 base types
typeof(sum)
is.primitive(sum)
```

![day1003](https://storage.googleapis.com/2017_ithome_ironman/day1003.png)

Base types 類別的物件會在 `typeof()` 函數回傳 **builtin**，在 `is.primitive()` 函數會被判斷為 `TRUE`。

### S3

S3 類別是 R 語言裡面最受歡迎的物件導向類別，內建的套件 `stats` 與 `base` 中全部都是使用 S3 類別。我們可以使用 `pryr::otype()` 函數來判斷某個物件是不是 S3 類別。

#### 建立一個 S3 物件

S3 物件不需要正式的宣告或預先定義，只要將一個 list 資料結構給一個類別名稱即可。

```{r}
library(pryr)

ironmen_list <- list(
    group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"),
    participants = c(51, 8, 18, 14, 6, 64)
)
class(ironmen_list) <- "ironmen"
ironmen_list
otype(ironmen_list)
```

![day1004](https://storage.googleapis.com/2017_ithome_ironman/day1004.png)

#### 屬性

使用 `$` 可以取得 S3 物件中的屬性（attributes），跟從 list 資料結構中依元素名稱選取的語法相同。

```{r}
ironmen_list <- list(
    group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"),
    participants = c(51, 8, 18, 14, 6, 64)
)
class(ironmen_list) <- "ironmen"

# 取得屬性
ironmen_list$group
ironmen_list$participants
```

![day1005](https://storage.googleapis.com/2017_ithome_ironman/day1005.png)

#### 方法

我們使用 `UseMethod()` 建立一個 S3 類別的方法 `count_participants` 來計算總鐵人數。

```{r}
# 建立一個 S3 物件
ironmen_list <- list(
    group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"),
    participants = c(51, 8, 18, 14, 6, 64)
)
class(ironmen_list) <- "ironmen"

# 建立方法
count_participants <- function(obj) {
    UseMethod("count_participants")
}
count_participants.ironmen <- function(obj) {
    return(sum(obj$participants))
}

# 呼叫方法
count_participants(ironmen_list)
```

![day1006](https://storage.googleapis.com/2017_ithome_ironman/day1006.png)

S3 物件的方法是內建函數（generic function）。

### S4

S4 類別相較於 S3 類別更加嚴謹。我們可以使用 `isS4()` 函數來判斷物件是不是 S4 類別。

#### 建立一個 S4 物件

我們需要使用 `setClass()` 函數來預先定義類別，設定有哪些 `slots` 以及他們的資料類型，並且使用 `new()` 函數建立新的物件。

```{r}
# 預先定義類別
setClass("ironmen", slots = list(group="character", participants = "numeric"))

# 建立 S4 物件
ironmen_list <- new("ironmen", group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))
ironmen_list
isS4(ironmen_list)
```

![day1007](https://storage.googleapis.com/2017_ithome_ironman/day1007.png)

#### 屬性

使用 `@` 可以取得 S4 物件中的屬性（attributes）。

```{r}
# 預先定義類別
setClass("ironmen", slots = list(group="character", participants = "numeric"))

# 建立 S4 物件
ironmen_list <- new("ironmen", group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))

# 取得屬性
ironmen_list@group
ironmen_list@participants
```

![day1008](https://storage.googleapis.com/2017_ithome_ironman/day1008.png)

#### 方法

我們使用 `setMethod()` 函數建立一個 S4 類別的方法 `count_participants` 來計算總鐵人數。

```{r}
# 預先定義類別
setClass("ironmen", slots = list(group="character", participants = "numeric"))

# 建立方法
setMethod("count_participants",
         "ironmen",
         function(obj) {
           return(sum(obj@participants))
         }
)

# 建立 S4 物件
ironmen_list <- new("ironmen", group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))

# 呼叫方法
count_participants(ironmen_list)
```

![day1009](https://storage.googleapis.com/2017_ithome_ironman/day1009.png)

S4 類別的方法跟 S3 類別的方法同樣是內建函數（generic function）。

### RC（Reference class）

RC 跟 Python 的物件導向較為相像，她建立出來的方法不是內建函數，而是屬於 RC 物件。

#### 建立一個 RC 物件

我們需要使用 `setRefClass()` 函數來預先定義，設定有哪些 `fields` 以及他們的資料類型。

```{r}
# 預先定義
Ironmen <- setRefClass("Ironmen", fields = list(group = "character", participants = "numeric"))

# 建立 RC 物件
ironmen_list <- Ironmen(group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))
ironmen_list
```

![day1010](https://storage.googleapis.com/2017_ithome_ironman/day1010.png)

#### 屬性

使用 `$` 可以取得 RC 物件中的屬性（attributes）。

```{r}
# 預先定義
Ironmen <- setRefClass("Ironmen", fields = list(group = "character", participants = "numeric"))

# 建立 RC 物件
ironmen_list <- Ironmen(group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))

ironmen_list$group
ironmen_list$participants
```

![day1011](https://storage.googleapis.com/2017_ithome_ironman/day1011.png)

#### 方法

在定義 RC 的時候將方法也撰寫進去。

```{r}
Ironmen <- setRefClass("Ironmen",
    fields = list(group = "character", participants = "numeric"),
    methods = list(
        count_participants = function() {
            return(sum(participants))
        }
    )
)
ironmen_list <- Ironmen(group = c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"), participants = c(51, 8, 18, 14, 6, 64))
ironmen_list$count_participants()
```

![day1012](https://storage.googleapis.com/2017_ithome_ironman/day1012.png)

### 比較表

我們用一張簡易的表格比較這三種物件導向類別。

|   |S3 類別|S4 類別|RC|
|---------------------|
|定義|沒有正式的宣告|使用 `setClass()` 宣告|使用 `setRefClass()` 宣告|
|使用屬性|使用 `$`|使用 `@`|使用 `$`|
|方法定義|`UseMethod()`|`setMethod()`|`setRefMethod()`|
|方法歸屬|屬於內建函數|屬於內建函數|屬於 RC 類別|

如果拿不定主意要選擇哪一個物件導向類別，就選擇 S3 類別吧！

> Three OO systems is a lot for one language, but for most R programming, S3 suffices. In R you usually create fairly simple objects and methods for pre-existing generic functions like print(), summary(), and plot(). S3 is well suited to this task, and the majority of OO code that I have written in R is S3. S3 is a little quirky, but it gets the job done with a minimum of code.
> [Hadley Wickham](http://hadley.nz/)

## 小結

第十天我們討論 R 語言的物件導向三大類別：S3 類別，S4 類別與 RC（Reference class），我們透過簡單的範例來新增類別，建立類別物件並且自己定義類別中的方法。

## 參考連結

- [Advanced R](http://adv-r.had.co.nz/)
- [PROGRAMIZ](https://www.programiz.com/)