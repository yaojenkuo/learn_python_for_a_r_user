# [第 07 天] 迴圈與流程控制

---

截至 2016-12-07 上午 11 時第 8 屆 iT 邦幫忙各組的鐵人分別是 49、8、12、12、6 與 61 人，我們想要在命令列上一一輸出這六組的鐵人數。

在不撰寫迴圈的情況下，我們還是可以先將這六個數字存在一個 R 語言的 vector 或 Python 的 list，然後再土法煉鋼依照索引值一一把結果輸出在命令列，如果你對這兩個資料結構有一點疑惑，我推薦你參考本系列文章的[[[第 04 天] 資料結構 List，Tuple 與 Dictionary](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day04.md)。

```
ironmen <- c(49, 8, 12, 12, 6, 61)
ironmen[1]
ironmen[2]
ironmen[3]
ironmen[4]
ironmen[5]
ironmen[6]
```

![day0701](https://storage.googleapis.com/2017_ithome_ironman/day0701.png)

```python
ironmen = [49, 8, 12, 12, 6, 61]
print(ironmen[0])
print(ironmen[1])
print(ironmen[2])
print(ironmen[3])
print(ironmen[4])
print(ironmen[5])
```

![day0702](https://storage.googleapis.com/2017_ithome_ironman/day0702.png)

好險現在鐵人賽只有六組，勉強我們還是可以土法煉鋼，但是如果有 66 組，666 組或者甚至 6,666 組的時候怎麼辦？當我們發現自己在進行複製貼上程式碼的時候，應該稍微把視線挪移開螢幕，想一下能怎麼樣把這段程式用迴圈取代。

## R 語言的迴圈

我們簡單使用 `for` 與 `while` 迴圈來完成任務。透過撰寫迴圈，不論 ironmen 這個 vector 紀錄了 6 組，66 組，666 組或者甚至 6,666 組的鐵人數，都是用同樣簡短的幾行程式即可。

### for 迴圈

```
# 不帶索引值的寫法
ironmen <- c(49, 8, 12, 12, 6, 61)
for (ironman in ironmen) {
    print(ironman)
}
ironman # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看

# 帶索引值的寫法
for (index in 1:length(ironmen)) {
    print(ironmen[index])
}
index # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0703](https://storage.googleapis.com/2017_ithome_ironman/day0703.png)

### while 迴圈

```
ironmen <- c(49, 8, 12, 12, 6, 61)
index <- 1
while (index <= length(ironmen)) {
    print(ironmen[index])
    index <- index + 1
}
index # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0704](https://storage.googleapis.com/2017_ithome_ironman/day0704.png)

在這裡稍微停一下，比較一下我們印出來的迭代器在這三種寫法中的差異。

## Python 的迴圈

Python 的迴圈撰寫結構類似 R 語言，不同的是R 語言使用大括號 `{}` 將每一次迭代要處理的事情包起來，Python 則是將每一次迭代要處理的事情以**縮排**表示。

### for 迴圈

```python
# 不帶索引值的寫法
ironmen = [49, 8, 12, 12, 6, 61]
for ironman in ironmen:
    print(ironman)
print("---")
print(ironman) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看

print("\n") # 空一行方便閱讀

# 帶索引值的寫法
for index in list(range(len(ironmen))): # 產生出一組 0 到 5 的 list
    print(ironmen[index])
print("---")
print(index) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0705](https://storage.googleapis.com/2017_ithome_ironman/day0705.png)

### while 迴圈

```python
ironmen = [49, 8, 12, 12, 6, 61]
index = 0
while index < len(ironmen):
    print(ironmen[index])
    index += 1
print("---")
print(index) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0706](https://storage.googleapis.com/2017_ithome_ironman/day0706.png)

在撰寫迴圈的時候你會發現到跟 R 語言因為索引值起始值不同（R 語言的資料結構索引值由 1 起始，Python 由 0 起始）而做出相對應的調整，另外，如果你對於 while 迴圈中 `index += 1` 的寫法感到陌生，我推薦你參考本系列文章的[[第 03 天] 變數類型的轉換](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day03.md)。

## R 語言的流程控制

R 語言可以透過 `if-else if-else` 的結構創造程式執行的分支，當我們只有兩個分支，使用 `if-else` 就足夠，多個分支時還有一個額外的 `switch()` 函數供我們選擇。

### if-else

整數除以 2 的餘數只會有兩種答案，使用 `if-else` 結構依照餘數的值回傳對應的訊息。

```
my_seq <- 1:10
for (i in my_seq) {
    if (i %% 2 == 0) {
        print(paste(i, "是偶數"))
    } else {
        print(paste(i, "是奇數"))
    }
}
```

![day0707](https://storage.googleapis.com/2017_ithome_ironman/day0707.png)

### if-else if-else

整數除以 3 的餘數會有三種答案，使用 `if-else if-else` 結構依照餘數的值回傳對應的訊息。

```
my_seq <- 1:10
for (i in my_seq) {
    if (i %% 3 == 0) {
        print(paste(i, "可以被 3 整除"))
    } else if (i %% 3 == 1) {
        print(paste(i, "除以 3 餘數是 1"))
    } else {
        print(paste(i, "除以 3 餘數是 2"))
    }
}
```

![day0708](https://storage.googleapis.com/2017_ithome_ironman/day0708.png)

### switch() 函數

將 `if-else if-else` 結構用 `switch()` 函數改寫，這裡要注意的是我們要先調整為用文字型態來判斷。

```
my_seq <- 1:10
for (i in my_seq) {
    ans <- i %% 3
    switch(as.character(ans),
        "0" = print(paste(i, "可以被 3 整除")),
        "1" = print(paste(i, "除以 3 餘數是 1")),
        "2" = print(paste(i, "除以 3 餘數是 2"))
    )
}
```

![day0709](https://storage.googleapis.com/2017_ithome_ironman/day0709.png)

## Python 的流程控制

Python 透過 `if-elif-else` 的結構創造程式執行的分支，當我們只有兩個分支，使用 `if-else` 就足夠，值得一提的是 Python 沒有類似像 `switch` 或是 `case` 的多分支結構。

### if-else

整數除以 2 的餘數只會有兩種答案，使用 `if-else` 結構依照餘數的值回傳對應的訊息。

```python
my_seq = list(range(1, 11))
for index in my_seq:
    if (index % 2 == 0):
        print(index, "是偶數")
    else:
        print(index, "是奇數")
```

![day0710](https://storage.googleapis.com/2017_ithome_ironman/day0710.png)

### if-elif-else

整數除以 3 的餘數會有三種答案，使用 `if-elif-else` 結構依照餘數的值回傳對應的訊息。

```python
my_seq = list(range(1,11))
for index in my_seq:
    if (index % 3 == 0):
        print(index, "可以被 3 整除")
    elif (index % 3 ==1):
        print(index, "除以 3 餘數是 1")
    else:
        print(index, "除以 3 餘數是 2")
```

![day0711](https://storage.googleapis.com/2017_ithome_ironman/day0711.png)

## 讓迴圈變得更有彈性

我們已經開始在迴圈中加入流程控制的結構，透過額外的描述，可以讓我們的迴圈在執行時更有彈性。

### R 語言的 break 與 next 描述

利用 `break` 描述告訴 for 迴圈當迭代器（此處指變數 ironman）小於 10 的時候要結束迴圈；利用 `next` 描述告訴 for 迴圈當迭代器小於 10 的時候要跳過它然後繼續執行。

```
# break 描述
ironmen <- c(49, 8, 12, 12, 6, 61)
for (ironman in ironmen) {
    if (ironman < 10) {
        break
    } else {
        print(ironman)
    }
}
ironman # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看

# next 描述
for (ironman in ironmen) {
    if (ironman <= 10) {
        next
    } else {
        print(ironman)
    }
}
ironman # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0712](https://storage.googleapis.com/2017_ithome_ironman/day0712.png)

### Python 的 break 與 continue

利用 `break` 描述告訴 for 迴圈當迭代器（此處指變數 ironman）小於 10 的時候要結束迴圈；利用 `continue` 描述告訴 for 迴圈當迭代器小於 10 的時候要跳過它然後繼續執行。

```python
# break 描述
ironmen = [49, 8, 12, 12, 6, 61]
for ironman in ironmen:
    if (ironman < 10):
        break
    else:
        print(ironman)

print("---")
print(ironman) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看

print("\n") # 空一行方便閱讀

# continue 描述
for ironman in ironmen:
    if (ironman < 10):
        continue
    else:
        print(ironman)

print("---")
print(ironman) # 把迴圈的迭代器（iterator）或稱游標（cursor）最後的值印出來看看
```

![day0713](https://storage.googleapis.com/2017_ithome_ironman/day0713.png)

## 小結

第七天我們討論 Python 的迴圈與流程控制，也與 R 語言相互對照，發現語法非常相似，我們需要記得 Python 的索引值起始值為 0 而非 1，使用縮排而非大括號 `{}` ，使用 `if-elif-else` 而非 `if-else if-else`，使用 `continue` 描述而非 `next`。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)