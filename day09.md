# [第 09 天] 函數（2）

---

一但體驗到自訂函數的威力與函數型編程的美觀，似乎開始有種**回不去**的感覺，好想要回頭把過去撰寫的每一段程式都拿來改寫成函數型編程的結構。我們再多花一天的時間研究函數，像是變數範圍（Scope），巢狀函數（Nested functions），錯誤處理（Error Handling）以及 Python 特殊的彈性參數（Flexible arguments）。

## 變數範圍（Scope）

開始自訂函數之後我們必須要意識到**變數範圍**這件事情，程式中的變數開始被我們區分為**區域變數（local variables）**與**全域變數（global variables）**。在函數內我們能夠使用兩種類型的變數，但是在函數外，僅能使用**全域變數**。用講的很抽象，我們還是動手定義自訂函數來釐清這個概念。

### R 語言

我們自訂一個單純的 `my_square()` 函數將輸入的數字平方後回傳。 

```
# 定義自訂函數
my_square <- function(input_number) {
    ans <- input_number^2 # 區域變數，只有在函數中可以被使用
    return(ans)
}

# 呼叫函數
my_square(3)

# 印出變數
ans # 無法印出區域變數
```

![day0901](https://storage.googleapis.com/2017_ithome_ironman/day0901.png)

換一種寫法我們將 `ans` 在函數外也指派一次。

```
ans <- 1 # 全域變數
# 定義自訂函數
my_square <- function(input_number) {
    ans <- input_number^2 # 區域變數，只有在函數中可以被使用
    return(ans)
}

# 呼叫函數
my_square(3)

# 印出變數
ans # 印出全域變數
```

![day0902](https://storage.googleapis.com/2017_ithome_ironman/day0902.png)

由這個例子我們可知 R 語言的函數會優先在**區域變數**尋找 `ans`，所以 `my_square(3)` 並沒有回傳 `1`。

### Python

我們自訂一個單純的 `my_square()` 函數將輸入的數字平方後回傳。 

```python
# 定義自訂函數
def my_square(input_number):
    '計算平方數'
    ans = input_number**2 # 區域變數，只有在函數中可以被使用
    return ans

# 呼叫函數
print(my_square(3))

# 印出變數
print(ans) # 無法印出區域變數
```

![day0903](https://storage.googleapis.com/2017_ithome_ironman/day0903.png)

換一種寫法我們將 `ans` 在函數外也指派一次。

```python
ans = 1 # 全域變數
# 定義自訂函數
def my_square(input_number):
    '計算平方數'
    ans = input_number**2 # 區域變數，只有在函數中可以被使用
    return ans

# 呼叫函數
print(my_square(3))

# 印出變數
print(ans) # 全域變數
```

![day0904](https://storage.googleapis.com/2017_ithome_ironman/day0904.png)

由這個例子我們可知 Python 的函數同樣會優先在**區域變數**尋找 `ans`，所以 `my_square(3)` 並沒有回傳 `1`。

## 巢狀函數（Nested functions）

我們可以在函數裡面嵌入函數，舉例來說一個計算平均數的函數裡面應該要包含兩個函數，一個是計算總和的函數 `my_sum()`，一個是計算個數的函數 `my_length()`。

### R 語言

```
# 定義自訂函數
my_mean <- function(input_vector) {
    my_sum <- function(input_vector) {
        temp_sum <- 0
        for (i in input_vector) {
            temp_sum <- temp_sum + i
        }
        return(temp_sum)
    }
    
    my_length <- function(input_vector) {
        temp_length <- 0
        for (i in input_vector) {
            temp_length <- temp_length + 1
        }
        return(temp_length)
    }
    return(my_sum(input_vector) / my_length(input_vector))
}

# 呼叫自訂函數
my_vector <- c(51, 8, 18, 13, 6, 64)
my_mean(my_vector)
```

![day0905](https://storage.googleapis.com/2017_ithome_ironman/day0905.png)

### Python

```python
# 定義自訂函數
def my_mean(input_list):
    '計算平均數'
    def my_sum(input_list):
        '計算總和'
        temp_sum = 0
        for i in input_list:
            temp_sum += i
        return temp_sum
    def my_length(input_list):
        '計算個數'
        temp_length = 0
        for i in input_list:
            temp_length += 1
        return temp_length
    return my_sum(input_list) / my_length(input_list)

# 呼叫自訂函數
my_list = [51, 8, 18, 13, 6, 64]
print(my_mean(my_list))
```

![day0906](https://storage.googleapis.com/2017_ithome_ironman/day0906.png)

## 錯誤處理（Error Handling）

我們在使用內建函數時候常有各種原因會導致錯誤或者警示，這時收到的回傳訊息可以幫助我們修改程式。

```
as.integer(TRUE)
as.integer("TRUE")
```

![day0907](https://storage.googleapis.com/2017_ithome_ironman/day0907.png)

```python
print(int(True))
print(int("True"))
```

![day0908](https://storage.googleapis.com/2017_ithome_ironman/day0908.png)

自訂函數時如果能夠掌握某些特定錯誤，撰寫客製的錯誤訊息，可以讓使用自訂函數的使用者更快完成偵錯。

### R 語言

R 語言使用 `tryCatch()` 函數進行錯誤處理，讓我們修改原本計算平方數的 `my_square()` 當使用者輸入文字時會回傳客製錯誤訊息：「請輸入數值。」

```
# 定義自訂函數
my_square <- function(input_number) {
    tryCatch({
        ans <- input_number^2
        return(ans)
        },
        error = function(e) {
            print("請輸入數值。")
        }
    )
}

# 呼叫自訂函數
my_square(3)
my_square('ironmen')
```

![day0909](https://storage.googleapis.com/2017_ithome_ironman/day0909.png)

### Python

Python 使用 `try - except` 的語法結構進行錯誤處理，讓我們修改原本計算平方數的 `my_square()` 當使用者輸入文字時會回傳客製錯誤訊息：「請輸入數值。」

```python
# 定義自訂函數
def my_square(input_number):
    '計算平方數且有錯誤處理的函數'
    try:
        ans = input_number**2
        return ans
    except:
        print("請輸入數值。")

# 呼叫自訂函數
print(my_square(3))
my_square('ironmen')
```

![day0910](https://storage.googleapis.com/2017_ithome_ironman/day0910.png)

## Python 的彈性參數（Flexible arguments）

Python 可以使用 `*args` 或 `**kwargs`(Keyword arguments)來分別處理不帶鍵值與帶有鍵值的彈性數量參數，利用這個特性，我們不一定要使用資料結構把變數包裝起來當做輸入。

### \*args

```python
# 定義自訂函數
def ironmen_list(*args):
    '列出各組參賽鐵人數'
    for ironman in args:
        print(ironman)

# 呼叫自訂函數
ironmen_list(51, 8, 18, 13, 6) # 不含自我挑戰組
print("---")
ironmen_list(51, 8, 18, 13, 6, 64) # 含自我挑戰組
```

![day0911](https://storage.googleapis.com/2017_ithome_ironman/day0911.png)

### \*\*kwargs

```python
# 定義自訂函數
def ironmen_list(**kwargs):
    '列出各組參賽鐵人數'
    for key in kwargs:
        print(key, "：", kwargs[key], "人")

ironmen_list(ModernWeb = 51, DevOps = 8, Cloud = 18, BigData = 13, Security = 6, 自我挑戰組 = 64)
```

![day0912](https://storage.googleapis.com/2017_ithome_ironman/day0912.png)

## 小結

第九天我們繼續討論 Python 的函數有關變數範圍，巢狀函數與錯誤處理的概念，並且跟 R 語言相互對照。在變數範圍與巢狀函數部分其實很相似；然而在錯誤處理上 Python 是使用 `try - except` 的語法結構，而 R 語言則是使用 `tryCatch()` 函數，這是比較大的差異處。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [Python Tutorial](https://www.tutorialspoint.com/python/index.htm)
- [Advanced R](https://www.amazon.com/dp/1466586966/ref=cm_sw_su_dp?tag=devtools-20)
- [Using R - Basic error Handing with tryCatch()](http://mazamascience.com/WorkingWithData/?p=912)