# [第 12 天] 常用屬性或方法 變數與基本資料結構

---

我們終於對物件（Objects）以及屬於她的屬性（Attributes）與方法（Methods）有了一定程度的瞭解，這對於非開發者背景的 R 語言使用者可是一個里程碑！還記得在前面的學習筆記中我們花了很多時間在討論 Python 不同的變數類型與資料結構嗎？既然我們已經建立了物件導向的概念，我們勢必也要熟悉如何在這些變數類型與資料結構上應用她們的屬性或者方法。

## 基本變數類型的屬性或方法

我們在 [[第 02 天] 基本變數類型](http://ithelp.ithome.com.tw/articles/10184855)討論過 Python 的基本變數類型（Built-in types），分為數值，文字與布林值三大類型，現在我們來瞭解這些變數類型可以應用的方法有哪些。

### 數值

#### float

浮點數（float）屬於 numbers.Real 類別（繼承自 abstract base 類別。）

- `as_integer_ratio()` 方法
- `is_integer()` 方法
- `hex()` 方法
- `fromhex()` 方法
- ...

```python
my_float = 8.7
print(my_float.as_integer_ratio())
print(my_float.is_integer())
print(my_float.hex())
print(float.fromhex("0x1.1666666666666p+3"))
```

![day1201](https://storage.googleapis.com/2017_ithome_ironman/day1201.png)

#### int

整數（int）屬於 numbers.Integral 類別（繼承自 abstract base 類別。）

- `bit_length()` 方法
- `to_bytes()` 方法
- `from_bytes()` 方法
- ...

```python
my_int = 87
print(my_int.bit_length())
print(my_int.to_bytes(length = 2, byteorder = "big"))
print(int.from_bytes(b'\x00W', byteorder = "big"))
print("---")
print(my_int.to_bytes(length = 10, byteorder = "big"))
print(int.from_bytes(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00W', byteorder = "big"))
```

![day1202](https://storage.googleapis.com/2017_ithome_ironman/day1202.png)

#### complex

複數（complex）屬於 numbers.Complex 類別（繼承自 abstract base 類別）。

- `real` 屬性
- `imag` 屬性
- `conjugate()` 方法

```python
my_complex = 8 + 7j
print(my_complex.real)
print(my_complex.imag)
print(my_complex.conjugate())
```

![day1203](https://storage.googleapis.com/2017_ithome_ironman/day1203.png)

### 文字（str）

文字（str）有太多方法可以使用，我們簡單列了一些常用方法。

- `startswith()` 方法
- `endswith()` 方法
- `find()` 方法
- `count()` 方法
- `strip()` 方法
- `capitalize()` 方法
- `title()` 方法
- `upper()` 方法
- `lower()` 方法
- `swapcase()` 方法
- `replace()` 方法
- ...

```python
my_str = "It's the 2017 ithelp ironman contest!!!"

print(my_str.startswith("It's")) # True
print(my_str.endswith("contest??")) # False
print(my_str.find("2017")) # 9
print(my_str.count("!")) # 3
print(my_str.strip("!")) # It's the 2017 ithelp ironman contest
print(my_str.capitalize()) # It's the 2017 ithelp ironman contest!!!
print(my_str.title()) # It'S The 2017 Ithelp Ironman Contest!!!
print(my_str.upper()) # IT'S THE 2017 ITHELP IRONMAN CONTEST!!!
print(my_str.lower()) # it's the 2017 ithelp ironman contest!!!
print(my_str.swapcase()) # iT'S THE 2017 ITHELP IRONMAN CONTEST!!!
print(my_str.replace("contest", "competition")) # It's the 2017 ithelp ironman competition!!!
```

![day1204](https://storage.googleapis.com/2017_ithome_ironman/day1204.png)

### 布林值（bool）

布林值（bool）的方法或屬性與 int 幾乎相同。

- `bit_length()` 方法
- `to_bytes()` 方法
- `from_bytes()` 方法
- ...

```python
my_bool = True
print(my_bool.bit_length())
print(my_bool.to_bytes(length = 2, byteorder = "big"))
print(bool.from_bytes(b'\x00\x01', byteorder = "big"))
print("---")
print(my_bool.to_bytes(length = 10, byteorder = "big"))
print(bool.from_bytes(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01', byteorder = "big"))
```

![day1205](https://storage.googleapis.com/2017_ithome_ironman/day1205.png)

## 基本資料結構的屬性或方法

我們在 [[第 04 天] 資料結構 List，Tuple 與 Dictionary](http://ithelp.ithome.com.tw/articles/10185010)討論過 Python 的基本資料結構（Built-in collections），分為 list，tuple 與 dict 三大類型，現在我們來瞭解這些資料結構可以應用的方法有哪些。

### list

list 有太多方法可以使用，我們簡單列了一些常用方法。

- `append()` 方法
- `insert()` 方法
- `remove()` 方法
- `pop()` 方法
- `index()` 方法
- `count()` 方法
- `sort()` 方法
- `reverse()` 方法
- ...

```python
# 截至 2016-12-12 下午 3 時第 8 屆 iT 邦幫忙各組的鐵人分別是 56、8、18、14、6 與 66 人
ironmen = [56, 8, 18, 14, 6]

ironmen.append(66)
print(ironmen)
ironmen.pop()
print(ironmen)
ironmen.insert(5, 66)
ironmen.remove(66)
print(ironmen)
ironmen.index(56)
ironmen.append(66)
ironmen.append(66)
print(ironmen.count(66))
ironmen.pop()
ironmen.sort()
print(ironmen)
ironmen.reverse()
print(ironmen)
```

![day1206](https://storage.googleapis.com/2017_ithome_ironman/day1206.png)

### tuple

tuple 是一個不可變（immutable）的資料結構，所以不會有改變內容的方法。

- `index()` 方法
- `count()` 方法

```python
my_tuple = (56, 8, 18, 14, 6, 6)
print(my_tuple.index(56))
print(my_tuple.count(6))
```

![day1207](https://storage.googleapis.com/2017_ithome_ironman/day1207.png)

### dictionary

dictionary 有太多方法可以使用，我們簡單列了一些常用方法。

- `get()` 方法
- `keys()` 方法
- `items()` 方法
- `values()` 方法
- ...

```python
ironmen_dict = {"Modern Web": 56,
                "DevOps": 8,
                "Cloud": 18,
                "Big Data": 14,
                "Security": 6,
                "自我挑戰組": 66
                }

print(ironmen_dict.get("Modern Web"))
print(ironmen_dict.keys())
print(ironmen_dict.items())
print(ironmen_dict.values())
```

![day1208](https://storage.googleapis.com/2017_ithome_ironman/day1208.png)

## 查看可以使用的屬性或方法

### 在 Jupyter Notebook 中以 tab 鍵查詢

當我們在 jupyter notebook 建立好物件（變數或資料結構）以後，可以在物件名稱後輸入 `.` 再按 **tab**鍵。

```python
ironmen_dict = {"Modern Web": 56,
                "DevOps": 8,
                "Cloud": 18,
                "Big Data": 14,
                "Security": 6,
                "自我挑戰組": 66
                }
```

![day1209](https://storage.googleapis.com/2017_ithome_ironman/day1209.png)

### 以 `dir()` 函數查詢

物件（變數或資料結構）建立好以後，可以使用 `dir()` 函數查詢。

```python
ironmen_dict = {"Modern Web": 56,
                "DevOps": 8,
                "Cloud": 18,
                "Big Data": 14,
                "Security": 6,
                "自我挑戰組": 66
                }

dir(ironmen_dict)
```

### 官方文件與 Google 搜尋

或者查閱 [Python 官方文件](https://docs.python.org/3/index.html)與 Google 搜尋。

## 小結

第十二天我們回顧了關於 Python 基本變數類型與資料結構可以應用的屬性或方法，在瞭解物件，屬性與方法的意義之後，我們可以很清楚地區別使用內建函數（generic functions），使用物件的屬性與呼叫物件的方法，在應用過程中我們也發現這些屬性或方法的命名大多數相當直觀，使用起來相當友善。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [Python 3.5.2 documentation](https://docs.python.org/3/index.html)