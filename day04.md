我們今天要從元素（ingredients）邁向結構（collections），這句話是什麼意思？如果你回顧一下前幾天的程式碼範例，會發現我們只有在物件中指派一個變數在裡面，然而不論是 R 語言或者是 Python，物件其實都像是一個巨大的箱子，我們當然可以很客氣地在裡頭指派一個變數，但也可以視需求善用箱子裡面的空間，比如我們可以用兩種方法儲存鐵人賽的組別資訊。

```{python}
# 方法一：客氣
ironman_group_1 = "Modern Web"
ironman_group_2 = "DevOps"
ironman_group_3 = "Cloud"
ironman_group_4 = "Big Data"
ironman_group_5 = "Security"
ironman_group_6 = "自我挑戰組"

# 方法二：善用
ironman_groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
```

`ironman_groups` 利用 Python 的資料結構 **list** 將六組文字儲存在一個物件之中。

在 R 語言我則會選擇用 **vector**。

```{r}
ironman_groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
```

接著我們看一下 R 語言的資料結構，然後再研究 Python。

## R 語言的資料結構

R 語言基本的資料結構大致有五類：

- vector
- factor
- matrix
- data frame
- list

我們可以透過使用 `factor()`，`matrix()`，`data.frame()`，`list()` 這些函數將原為**vector**結構的 `ironman_groups` 轉換為不同的資料結構。

```{r}
ironman_groups <- c("Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組")
# 轉換為不同的資料結構
ironman_groups_factor <- factor(ironman_groups)
ironman_groups_matrix <- matrix(ironman_groups, nrow = 2)
ironman_groups_df <- data.frame(ironman_groups)
ironman_groups_list <- list(ironman_groups, ironman_groups_factor, ironman_groups_matrix, ironman_groups_df)

# 印出這些資料結構
ironman_groups
ironman_groups_factor
ironman_groups_matrix
ironman_groups_df
ironman_groups_list
```

![day0401](https://storage.googleapis.com/2017_ithome_ironman/day0401.png)

![day0402](https://storage.googleapis.com/2017_ithome_ironman/day0402.png)

在 R 語言中，vector 與 factor 是處理僅有一個維度的資料結構，matrix 與 data frame 是處理有兩個維度，即列（row）與欄（column）的資料結構，而 list 就像是一個超級容器，它能夠容納不同資料結構的物件，我們可以透過兩個中括號 `[[]]` 搭配索引值選擇被容納於其中的物件，R 語言的索引值由 1 開始，這跟 Python 索引值由 0 開始有很大的區別。

```{r}
ironman_groups_list[[1]]
ironman_groups_list[[2]]
ironman_groups_list[[3]]
ironman_groups_list[[4]]
```

![day0403](https://storage.googleapis.com/2017_ithome_ironman/day0403.png)

## Python 的資料結構

Python 需要透過兩個套件 `numpy` 與 `pandas` 來產出能夠與 R 語言相互對應的資料結構（如 matrix 與data frame ... 等。）我們今天先不談需要引用套件的部分，留待後幾天再來討論。

Python 基本的資料結構大致有三類：

- list
- tuple
- dictionary

### list

Python 的 list 跟 R 語言的 list 相似，可以容納不同的變數類型與資料結構，雖然它的外觀長得跟 R 語言的 vector 比較像，但是他們之間有著截然不同的特性，那就是 R 語言的 vector 會將元素轉換成為同一變數類型，但是 list 不會，我們參考一下底下這個簡單的範例。

```{r}
participated_group <- "Big Data"
current_ttl_articles <- 4
is_participating <- TRUE

# 建立一個 vector
my_vector <- c(participated_group, current_ttl_articles, is_participating)
class(my_vector[1])
class(my_vector[2])

# 建立一個 list
my_list <- list(participated_group, current_ttl_articles, is_participating)
class(my_list[[1]])
class(my_list[[2]])
```

![day0404](https://storage.googleapis.com/2017_ithome_ironman/day0404.png)

在建立 Python 的 list 時候我們只需要使用中括號 `[]` 將元素包起來，而在選擇元素也是使用中括號 `[]` 搭配索引值，Python 的索引值由 0 開始，這跟 R 語言索引值由 1 開始有很大的區別。

```{python}
participated_group = "Big Data"
current_ttl_articles = 4
is_participating = True

my_status = [participated_group, current_ttl_articles, is_participating]
print(type(my_status[0]))
print(type(my_status[1]))
print(type(my_status[2]))
```

![day0405](https://storage.googleapis.com/2017_ithome_ironman/day0405.png)

### tuple

tuple 跟 list 很像，但是我們不能新增，刪除或者更新 tuple 的元素，這樣的資料結構沒有辦法對應到 R 語言。我們可以使用 `tuple()` 函數將既有的 list 轉換成為 tuple，或者在建立物件的時候使用小括號 `()` 有別於建立 list 的時候使用的中括號 `[]`。

```{python}
# 分別建立 list 與 tuple
ironman_groups_list = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security"]
ironman_groups_tuple = tuple(ironman_groups_list)

# 新增一個元素
ironman_groups_list.insert(5, "自我挑戰組")
ironman_groups_tuple.insert(5, "自我挑戰組")
```

![day0406](https://storage.googleapis.com/2017_ithome_ironman/day0406.png)

我們得到一個錯誤訊息，但如果我們將這段程式的最後一行註解掉，就沒有問題。

![day0407](https://storage.googleapis.com/2017_ithome_ironman/day0407.png)

在這段程式碼中你可能對於使用 `.insert()` 這個方法的寫法感到陌生，請暫時先不要理會它，在這裡我們只是要展示 list 是可以新增元素但是 tuple 不可以。

### dictionary

dictionary 是帶有鍵值（key）的 list，這樣的特性讓我們在使用中括號 `[]` 選擇元素時可以使用鍵值，在建立 Python 的 dictionary 時候我們只需要使用大括號 `{}` 或者使用 `dict()` 函數轉換既有的 list。

```{python}
participated_group = "Big Data"
current_ttl_articles = 4
is_participating = True

# 建立 dictionary
my_status = {
    "group": participated_group,
    "ttl_articles": current_ttl_articles,
    "is_participating": is_participating
}

# 利用鍵值選擇元素
print(my_status["group"])
print(my_status["ttl_articles"])
print(my_status["is_participating"])
```

![day0408](https://storage.googleapis.com/2017_ithome_ironman/day0408.png)

在 R 語言中建立 list 時進行命名，也可以生成類似 dictionary 的資料結構。

```{r}
participated_group <- "Big Data"
current_ttl_articles <- 4
is_participating <- TRUE

# 建立 named list
my_status <- list(
    group = participated_group,
    ttl_articles = current_ttl_articles,
    is_participating = is_participating
)

# 利用名稱選擇元素
my_status[["group]]
my_status[["ttl_articles"]]
my_status[["is_participating"]]
```

## 小結

第四天我們正式從元素（ingredients）邁向結構（collections），研究 Python 的基本資料結構，並且跟 R 語言的基本資料結構相互對照，Python 的 list 跟 R 語言的 list 可能外觀上看起來不太像，但是她們都可以容納不同的變數類型與資料結構。我們在後面幾天的文章也會探討 `numpy` 與 `pandas` 套件的資料結構應用。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [Python For Data Science Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)