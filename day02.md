# [R 語言使用者的 Python 學習筆記 - 第 02 天] 基本變數類型

---

暸解變數類型是學習程式語言的基本功之一（其他像是資料結構，流程控制或者迭代語法也歸類於基本功），它的枯燥無味常讓初學者為之怯步，因此有很多人會喜歡先從應用端：像是資料框，視覺化或者機器學習套件切入，之後再找時間回頭扎根。我其實很鼓勵在時間緊迫的前提下採取速成作法，但是既然 iT 邦幫忙給了我們連續三十天的餘裕，我認為還是應當把基本功的部分放在前面。

我們先看一下 R 語言的基本變數類型，然後再研究 Python。

## R 語言的基本變數類型

R 語言的基本變數類型分為以下這幾類：

- 數值
    - numeric
    - integer
    - complex
- 邏輯值（logical）
- 文字（character）

R 語言回傳變數類型的函數是 `class()`，如果不清楚這個函數有要如何使用，你可以打開 RStudio 然後在命令列輸入 `?class` 或者 `help(class)` 來看說明文件。

![day0201](https://storage.googleapis.com/2017_ithome_ironman/day0201.png)

我們依序在 RStudio 的命令列輸入下列指令就可以看到變數類型被回傳。 

```{r}
class(5) # "numeric"
class(5.5) # "numeric"
class(5L) # "integer"
class(5 + 3i) # "complex"
class(TRUE) # "logical"
class(FALSE) # "logical"
class("2017 ithome ironman") # "character"
```

![day0202](https://storage.googleapis.com/2017_ithome_ironman/day0202.png)

那麼關於 Python 的變數類型呢？是否也能夠與 R 語言相互對應？

## Python 的基本變數類型

從終端機開啟 Jupyter Notebook 然後新增一個 Python 3 的 Notebook。如果你對這段文字有些疑問，我推薦你看第一天的學習筆記：[建立開發環境與計算機應用](http://ithelp.ithome.com.tw/articles/10184561)。

Python 的基本變數類型分為以下這幾類：

- 數值
    - float
    - int
    - complex
- 布林值（bool）
- 文字（str）

Python 回傳變數類型的函數是 `type()`，如果不清楚這個函數有哪些參數可以使用，你可以在 cell 中輸入 `help(type)` 來看說明文件。

![day0203](https://storage.googleapis.com/2017_ithome_ironman/day0203.png)

我們在 cell 輸入下列指令然後執行印出變數類型，如果你不清楚怎麼執行 cell 中的指令，我同樣也推薦你看第一天的學習筆記：[建立開發環境與計算機應用](http://ithelp.ithome.com.tw/articles/10184561)。

```{python}
print(type(5)) # 'int'
print(type(5.5)) # 'float'
print(type(5 + 3j)) # 'complex'
print(type(True)) # 'bool'
print(type(False)) # 'bool'
print(type("2017 ithome ironman")) # 'str'
```

![day0204](https://storage.googleapis.com/2017_ithome_ironman/day0204.png)

跟 R 語言不同的地方是 Python 會自動區別 int 與 float，複數的宣告使用 `j` 而不是 `i`，布林值使用 `True/False` 而不是 `TRUE/FALSE`。當然，兩者使用的變數類型名稱也有所區別，但是我們可以看到大抵是都能夠很直觀地相互對應，例如：character 對應 str，然後 logical 對應 bool，以及 numeric 對應 float。

## 不同變數類型之間的運算

R 語言以 1 儲存 `TRUE`，0 儲存 `FALSE`，所以我們可以彈性地運算數值和邏輯值，但是文字就不能夠彈性地運算。

```{r}
1 == TRUE # TRUE
0L == FALSE # TRUE
1.2 + TRUE # 2.2
3L + TRUE * 2 # 5
"2017 ithome ironman" + " rocks!" # Error
```

![day0205](https://storage.googleapis.com/2017_ithome_ironman/day0205.png)

Python 儲存布林值的方式與 R 語言相同，因此也可以彈性地運算數值和布林值，除此以外，在文字上運算的彈性較 R 語言更大一些，可以利用 `+` 進行合併，以及利用 `*` 進行複製。

```{python}
print(1.0 == True) # True
print(0 == False) # True
print(1.2 + True) # 2.2
print(3 + True * 2) # 5
print("2017 ithome ironman" + " rocks!") # "2017 ithome ironman rocks!"
print("2017 ithome ironman " + "rocks" + "!" * 3) # "2017 ithome ironman rocks!!!"
```

![day0206](https://storage.googleapis.com/2017_ithome_ironman/day0206.png)

## 小結

第二天我們介紹了 Python 的基本變數類型，以及它們與 R 語言基本變數類型之間的對應，然後測試了不同變數類型之間運算的彈性。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)