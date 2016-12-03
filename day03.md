# [R 語言使用者的 Python 學習筆記 - 第 03 天] 變數類型的轉換

---

不同的變數類型之間並不是壁壘分明，井水不犯河水，比如在 Python 中 `True/False` 在與數值作加減運算的時候就會自動地被轉換為 `1` 與 `0`，然而在一些不是這麼顯而易見的情況之下，就要仰賴手動進行變數類型的轉換，比方說我想要 Python 印出下列這句話會得到 TypeError。

```{python}
days = 30
print("In order to become an ironman, you have to publish an article a day for " + days + " days in a row.")
```

![day0301](https://storage.googleapis.com/2017_ithome_ironman/day0301.png)

## 建立物件

我們在這段程式已經開始建立物件，在 Python 中將變數指派給物件的運算子是慣用的 `=`，跟 R 語言慣用的 `<-` 有所區別，當然 R 語言也接受使用 `=` 作為指派的運算子，但是在寫作風格上絕大多數的 R 語言使用者還是偏愛 `<-`。

Python 具備很便利的指派運算子像是：`+=`，`-=`，`*=`，`/=`，`%/` 讓我們的程式更簡潔，像是這段程式：

```{python}
days = 30
days = days + 3
days # 33
```

![day0305](https://storage.googleapis.com/2017_ithome_ironman/day0305.png)

其中的 `days = days + 3` 可以寫作 `days += 3`。

```{python}
days = 30
days += 3
days # 33
```

![day0306](https://storage.googleapis.com/2017_ithome_ironman/day0306.png)

這些指派運算子在 R 語言是沒有辦法使用的，所以這樣的寫法其實對我而言是陌生的，所以我多寫了幾行感受一下。

```{python}
days = 30
days += 3
print(days) # 33
days -= 3
print(days) # 30
days *= 5
print(days) # 150
days /= 5
print(days) # 30.0
days %= 7
print(days) # 2.0
```

![day0307](https://storage.googleapis.com/2017_ithome_ironman/day0307.png)

練習了建立物件後，接著我們回歸正題，看一下 R 語言如何轉換變數類型，然後再研究 Python。

## R 語言變數類型的轉換

在 R 語言透過 `paste()` 函數不需要做變數類型的轉換就可以完成在 Python 得到 TypeError 的那個例子。

```{r}
days <- 30
paste("In order to become an ironman, you have to publish an article a day for", days, "days in a row.")
```

![day0302](https://storage.googleapis.com/2017_ithome_ironman/day0302.png)

R 語言轉換變數類型的函數都是以 `as.` 作為開頭然後將要轉換為的變數類型接於後面，方便我們記憶。

- `as.numeric()`：轉換變數類型為 numeric
- `as.integer()`：轉換變數類型為 integer
- `as.complex()`：轉換變數類型為 complex
- `as.logical()`：轉換變數類型為 logical
- `as.character()`：轉換變數類型為 character

我們利用最有彈性的邏輯值來展示這幾個函數的功能。

```{r}
my_logical <- TRUE
class(my_logical) # "logical"
as.numeric(my_logical) # 1
as.integer(my_logical) # 1
as.complex(my_logical) # 1+0i
as.character(my_logical) # "TRUE"
```

![day0303](https://storage.googleapis.com/2017_ithome_ironman/day0303.png)

轉換變數類型的函數也不是萬能，比如說 `as.integer("TRUE")` 不會成功，想要將 `"TRUE"` 轉換為整數就要使用兩次轉換類型的函數 `as.integer(as.logical("TRUE"))`。

## Python 變數類型的轉換

透過 `str()` 函數就可以修正先前碰到的 TypeError 問題。

```{python}
days = 30
print("In order to become an ironman, you have to publish an article a day for " + str(days) + " days in a row.")
```

![day0304](https://storage.googleapis.com/2017_ithome_ironman/day0304.png)

Python 轉換變數類型的函數：

- `float()`：轉換變數類型為 float
- `int()`：轉換變數類型為 int
- `complex()`：轉換變數類型為 complex
- `bool()`：轉換變數類型為 bool
- `str()`：轉換變數類型為 str

我們利用最有彈性的布林值來展示這幾個函數的功能。

```{python}
my_bool = True
print(type(my_bool)) # 'bool'
print(float(my_bool)) # 1.0
print(int(my_bool)) # 1
print(complex(my_bool)) # 1+0j
print(type(str(my_bool))) # 'str'
```

![day0308](https://storage.googleapis.com/2017_ithome_ironman/day0308.png)

跟 R 語言相同，轉換變數類型的函數也不是萬能，比如說 `int("True")` 不會成功，想要將 `"True"` 轉換為整數就要使用兩次轉換類型的函數 `int(bool("True"))`。

## 小結

第三天我們藉由練習 Python 的指派運算子暖身，然後研究 Python 轉換變數類型的函數，並且跟 R 語言轉換變數類型的函數相互對照。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)