# [R 語言使用者的 Python 學習筆記 - 第 08 天] 函數

---

早在 [[R 語言使用者的 Python 學習筆記 - 第 02 天] 基本變數類型](http://ithelp.ithome.com.tw/articles/10184855)我們就開始使用 Python 的內建函數（generic functions），像是我們使用 `help()` 函數查詢文件以及使用 `type()` 函數來觀察我們的變數類型為 str，int 或者 bool。對於 R 語言使用者而言，函數可是我們的最愛，因為本質上 R 語言是一個函數型編程語言（functional programming language）。

> (Almost) everything is a function call.
> By [John Chambers](https://en.wikipedia.org/wiki/John_Chambers_(statistician))

## 應用內建函數

截至 2016-12-08 上午 8 時第 8 屆 iT 邦幫忙各組的鐵人分別是 50、8、16、12、6 與 62 人，我們將這個資訊存在一個 R 語言的 vector 或 Python 的 list，然後對這個資料結構使用一些內建函數。如果你對這兩個資料結構有點疑惑，我推薦你參考本系列文章的 [[R 語言使用者的 Python 學習筆記 - 第 04 天] 資料結構](http://ithelp.ithome.com.tw/articles/10185010)。

```{r}
ironmen <- c(50, 8, 16, 12, 6, 62)

# 應用函數
max(ironmen) # 最多的鐵人數
min(ironmen) # 最少的鐵人數
length(ironmen) # 總共有幾組
sort(ironmen) # 遞增排序
sort(ironmen, decreasing = TRUE) # 遞減排序
```

![day0801](https://storage.googleapis.com/2017_ithome_ironman/day0801.png)

```{python}
ironmen = [50, 8, 16, 12, 6, 62]

# 應用函數
print(max(ironmen)) # 最多的鐵人數
print(min(ironmen)) # 最少的鐵人數
print(len(ironmen)) # 總共有幾組
print(sorted(ironmen)) # 遞增排序
print(sorted(ironmen, reverse = True)) # 遞減排序
```

![day0802](https://storage.googleapis.com/2017_ithome_ironman/day0802.png)

## 查詢函數文件

不論是 R 語言或者 Python，我們都必須要求自己養成查詢函數文件的習慣，了解一個函數的輸入與它可以設定的參數，例如上面例子中，查詢文件之後就知道能在排序的函數中修改參數調整為遞增或遞減排序。

```{r}
?sort
help(sort)
```

![day0803](https://storage.googleapis.com/2017_ithome_ironman/day0803.png)

```{python}
sorted?
help(sorted)
```

![day0804](https://storage.googleapis.com/2017_ithome_ironman/day0804.png)

為了讓程式更加有組織性，更好管理與具備重複使用性，除了使用內建函數以外我們得要學習自訂函數。在自訂函數時候我們會使用**迴圈**與**流程控制**，如果你對這兩個概念有點疑惑，我推薦你參考本系列文章的 [[R 語言使用者的 Python 學習筆記 - 第 07 天] 迴圈與流程控制](http://ithelp.ithome.com.tw/articles/10185299)。

## R 語言的自訂函數

R 語言自訂函數的架構：

```{r}
function_name <- function(輸入, 參數 1, 參數 2, ...) {
    # 函數做些什麼事
    return(結果)
}
```

我們利用兩個練習來熟悉如何自訂函數。

### 計算圓形的面積或周長

第一個練習是輸入圓形的半徑長，依照參數的指定回傳面積或周長。

```{r}
# 定義自訂函數
circle_calculate <- function(radius, area = TRUE) {
    circle_area <- pi * radius^2
    circle_circum <- 2 * pi * radius
    if (area == TRUE) {
        return(circle_area)
    } else {
        return(circle_circum)
    }
}

# 呼叫自訂函數
my_radius <- 3
circle_calculate(my_radius) # 預設回傳面積
circle_calculate(my_radius, area = FALSE) # 指定參數回傳周長
```

![day0805](https://storage.googleapis.com/2017_ithome_ironman/day0805.png)

### 交換排序法（exchange sort）

第二個練習是寫程式的基本功交換排序法。

```{r}
# 定義自訂函數
exchange_sort <- function(input_vector, decreasing = FALSE) {
    input_vector_cloned <- input_vector # 複製一個輸入向量
    # 遞增排序
    if (decreasing == FALSE) {
        for (i in 1:(length(input_vector) - 1)) {
            for (j in (i + 1):length(input_vector)) {
               # 如果前一個數字比後一個數字大則交換位置
               if (input_vector_cloned[i] > input_vector_cloned[j]) {
                   temp <- input_vector_cloned[i]
                   input_vector_cloned[i] <- input_vector_cloned[j]
                   input_vector_cloned[j] <- temp
               }
            }
        }
    # 遞減排序
    } else {
        for (i in 1:(length(input_vector) - 1)) {
            for (j in (i + 1):length(input_vector)) {
               # 如果前一個數字比後一個數字小則交換位置
               if (input_vector_cloned[i] < input_vector_cloned[j]) {
                   temp <- input_vector_cloned[i]
                   input_vector_cloned[i] <- input_vector_cloned[j]
                   input_vector_cloned[j] <- temp
               }
            }
        } 
    }
    return(input_vector_cloned)
}

# 呼叫自訂函數
my_vector <- round(runif(10) * 100) # 產生一組隨機數
my_vector # 看看未排序前
exchange_sort(my_vector) # 預設遞增排序
exchange_sort(my_vector, decreasing = TRUE) # 指定參數遞減排序
```

![day0806](https://storage.googleapis.com/2017_ithome_ironman/day0806.png)

## Python 的自訂函數

Python 自訂函數的架構：

```{python}
def function_name(輸入, 參數 1, 參數 2, ...):
    '''
    Docstrings
    '''
    # 函數做些什麼事
    return 結果
```

Python 使用者習慣加入 Docstrings 做自訂函數的說明，接著我們利用兩個練習來熟悉如何自訂函數。

### 計算圓形的面積或周長

第一個練習是輸入圓形的半徑長，依照參數的指定回傳面積或周長。

```{python}
import math # 要使用 pi 得引入套件 math

# 定義自訂函數
def circle_calculate(radius, area = True):
    '依據輸入的半徑與 area 參數計算圓形的面積或周長' # 單行的 docstring
    circle_area = math.pi * radius**2
    circle_circum = 2 * math.pi * radius
    if area == True:
        return circle_area
    else:
        return circle_circum

# 呼叫自訂函數
my_radius = 3
print(circle_calculate(my_radius)) # 預設回傳面積
print(circle_calculate(my_radius, area = False)) # 指定參數回傳周長
```

![day0807](https://storage.googleapis.com/2017_ithome_ironman/day0807.png)

### 交換排序法（exchange sort）

第二個練習是寫程式的基本功交換排序法。

```{python}
import random # 呼叫函數時使用隨機整數

# 定義自訂函數
def exchange_sort(input_list, reverse = False):
    ''' # 多行的 docstrings
    依據輸入的 list 與 reverse 參數排序 list 中的數字後回傳。
    reverse 參數預設為 False 遞增排序，可以修改為 True 遞減排序。
    '''
    input_list_cloned = input_list
    # 遞增排序
    if reverse == False:
        for i in range(0, len(input_list) - 1):
            for j in range(i+1, len(input_list)):
                # 如果前一個數字比後一個數字大則交換位置
                if input_list_cloned[i] > input_list_cloned[j]:
                    temp = input_list_cloned[i]
                    input_list_cloned[i] = input_list_cloned[j]
                    input_list_cloned[j] = temp
    # 遞減排序
    else:
        for i in range(0, len(input_list) - 1):
            for j in range(i+1, len(input_list)):
                # 如果前一個數字比後一個數字小則交換位置
                if input_list_cloned[i] < input_list_cloned[j]:
                    temp = input_list_cloned[i]
                    input_list_cloned[i] = input_list_cloned[j]
                    input_list_cloned[j] = temp
    return input_list_cloned

# 呼叫自訂函數
my_vector = random.sample(range(0, 100), 10) # 產生一組隨機數
print(my_vector) # 看看未排序前
print(exchange_sort(my_vector)) # 預設遞增排序
print(exchange_sort(my_vector, reverse = True)) # 指定參數遞減排序
```

![day0808](https://storage.googleapis.com/2017_ithome_ironman/day0808.png)

## 使用自訂函數回傳多個值

### R 語言自訂函數回傳多個值

使用 list 資料結構將回傳值包裝起來再依名稱呼叫。

```{r}
# 定義自訂函數
ironmen_stats <- function(ironmen_vector) {
    max_ironmen <- max(ironmen_vector)
    min_ironmen <- min(ironmen_vector)
    ttl_groups <- length(ironmen_vector)
    ttl_ironmen <- sum(ironmen_vector)
    
    stats_list <- list(max_ironmen = max_ironmen,
                       min_ironmen = min_ironmen,
                       ttl_groups = ttl_groups,
                       ttl_ironmen = ttl_ironmen
                       )
    
    return(stats_list)
}

# 呼叫自訂函數
ironmen <- c(50, 8, 16, 12, 6, 62)
paste("最多：", ironmen_stats(ironmen)$max_ironmen, sep = '')
paste("最少：", ironmen_stats(ironmen)$min_ironmen, sep = '')
paste("總組數：", ironmen_stats(ironmen)$ttl_groups, sep = '')
paste("總鐵人數：", ironmen_stats(ironmen)$ttl_ironmen, sep = '')
```

![day0809](https://storage.googleapis.com/2017_ithome_ironman/day0809.png)

### Python 自訂函數回傳多個值

在 `return` 後面將多個值用逗號 `,` 隔開就會回傳一個 tuple。

```{python}
# 定義自訂函數
def ironmen_stats(ironmen_list):
    max_ironmen = max(ironmen_list)
    min_ironmen = min(ironmen_list)
    ttl_groups = len(ironmen_list)
    ttl_ironmen = sum(ironmen_list)
    return max_ironmen, min_ironmen, ttl_groups, ttl_ironmen
    
# 呼叫自訂函數
ironmen = [50, 8, 16, 12, 6, 62]
max_ironmen, min_ironmen, ttl_groups, ttl_ironmen = ironmen_stats(ironmen)
print("\n", 最多：", max_ironmen, "\n", "最少：", min_ironmen, "\n", "總組數：", ttl_groups, "\n", "總鐵人數：", ttl_ironmen)
```

![day0810](https://storage.googleapis.com/2017_ithome_ironman/day0810.png)

## 匿名函數

我們常常懶得為一個簡單的函數取名字，R 語言與 Python 很貼心都支援匿名函數。

### R 語言的匿名函數

利用匿名函數 `function(x) x * 30` 把每組的鐵人數乘以 30 可以得到預期的各組文章總數。

```{r}
ironmen <- c(50, 8, 16, 12, 6, 62)
ironmen_articles <- sapply(ironmen, function(x) x * 30)
ironmen_articles
```

![day0811](https://storage.googleapis.com/2017_ithome_ironman/day0811.png)

### Python 的 lambda 函數

Python 的匿名函數稱為 lambda 函數，利用 `lambda x : x * 30` 把每組的鐵人數乘以 30 可以得到預期的各組文章總數。

```{python}
ironmen = [50, 8, 16, 12, 6, 62]
ironmen_articles = list(map(lambda x : x * 30, ironmen))
print(ironmen_articles)
```

![day0812](https://storage.googleapis.com/2017_ithome_ironman/day0812.png)

## 小結

第八天我們討論 Python 的函數，也與 R 語言相互對照，發現在使用內建函數時的語法大同小異；至於在自訂函數的結構上，我們需要記得 Python 有撰寫 Docstrings 的好習慣，使用縮排而非大括號 `{}` 區隔參數與函數主體，可以較簡便地回傳多個值，以及使用 lambda 關鍵字表示使用匿名函數。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [Advanced R by Hadley Wickham](https://www.amazon.com/dp/1466586966/ref=cm_sw_su_dp?tag=devtools-20)
- [Rajath Kumar M P's Python-Lectures](https://github.com/rajathkumarmp/Python-Lectures)