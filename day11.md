# [R 語言使用者的 Python 學習筆記 - 第 11 天] 物件導向（2）

---

> Everything in Python, from numbers to modules, is an object.
> [Bill Lubanovic](http://www.oreilly.com/pub/au/2909)

經歷 R 語言令人困惑且有點崩潰的三個物件導向類別（S3 類別，S4 類別與 RC）後，我們回到標準的物件導向語言 Python。這裡提供一個主觀判斷（Judgement call）：習慣**函數型編程**的資料科學初學者應該先學 R 語言，而習慣**物件導向編程**的資料科學初學者應該先學 Python。但這個主觀判斷仍舊不能廣泛應用，因為這對於沒有接觸過任何一種類型編程的資料科學初學者來說毫無參考價值。

我們在開始討論 Python 物件導向之前再看一個熟悉的例子，藉此瞭解屬性與方法是什麼。

## 屬性與方法

一個**物件（Object）**可以包含**屬性（Attribute）**與**方法（Method）**，我們示範的物件是一個 data frame 叫做 `ironmen_df`，她有 `dtypes` 屬性與 `head()` 方法。

```{python}
import pandas as pd # 引用套件並縮寫為 pd

groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
# 截至 2016-12-11 上午 11 時第 8 屆 iT 邦幫忙各組的鐵人分別是 54、8、18、14、6 與 65 人
ironmen = [54, 8, 18, 14, 6, 65]

ironmen_dict = {"groups": groups,
                "ironmen": ironmen
                }

ironmen_df = pd.DataFrame(ironmen_dict)
print(ironmen_df.dtypes) # ironmen_df 有 dtypes 屬性
ironmen_df.head(n = 3) # ironmen_df 有 head() 方法
```

![day1101](https://storage.googleapis.com/2017_ithome_ironman/day1101.png)

在 R 語言中，最後兩行程式的處理我們都會使用函數 `class()` 與 `head()`，因此對 R 語言者而言，在學習 Python 物件導向時會需要多花一點心力理解屬性與方法。

## Python 的物件導向

### 定義類別（Class）

我們使用 `class` 語法來定義類別，並使用大寫開頭（Capitalized）單字為類別命名，如果對於 `__init__` 方法與 `self` 參數感到困惑，就先記得這是一個特殊的 Python 方法，它用來幫助我們創造屬於這個類別的物件。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants

print(Ironmen)
```

![day1102](https://storage.googleapis.com/2017_ithome_ironman/day1102.png)

### 根據類別建立物件（Object）

一但類別 `Ironmen` 被定義完成，就可以使用 `Ironmen()` 當作建構子（Constructor）建立物件。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants

# 根據 Ironmen 類別建立一個物件 modern_web
modern_web = Ironmen("Modern Web", 54)
print(modern_web)
```

![day1103](https://storage.googleapis.com/2017_ithome_ironman/day1103.png)

### 使用物件的屬性（Attribute）

在物件名稱後面使用 `.` 接屬性名稱就可以使用。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants

# 根據 Ironmen 類別建立一個物件 modern_web
modern_web = Ironmen("Modern Web", 54)

# 印出 modern_web 的兩個屬性
print(modern_web.group)
print(modern_web.participants)

# 印出 modern_web 的類別 doc string
print(modern_web.__doc__)
```

![day1104](https://storage.googleapis.com/2017_ithome_ironman/day1104.png)

在我們建立好屬於 `Ironmen` 類別的 `modern_web` 物件後，在 jupyter notebook 中我們還可以透過 **tab 鍵**來查看這個類別有哪些屬性（前後帶有兩個底線 `__` 的是預設屬性。）

![day1105](https://storage.googleapis.com/2017_ithome_ironman/day1105.png)

![day1106](https://storage.googleapis.com/2017_ithome_ironman/day1106.png)

我們也可以使用內建函數 `dir()` 來列出物件全部的屬性。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants

# 根據 Ironmen 類別建立一個物件 modern_web
modern_web = Ironmen("Modern Web", 54)

# 使用 dir() 函數
dir(modern_web)
```

![day1107](https://storage.googleapis.com/2017_ithome_ironman/day1107.png)

### 定義方法（Method）

除了 `__init__` 方法之外利用 `def` 定義更多屬於這個類別的方法。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants
    
    def print_info(self):
        print(self.group, "組有", self.participants, "位鐵人參賽！")

# 根據 Ironmen 類別建立一個物件 modern_web
modern_web = Ironmen("Modern Web", 54)

# 根據 Ironmen 類別建立一個物件 dev_ops
dev_ops = Ironmen("DevOps", 8)

# 使用 modern_web 的 print_info() 方法
modern_web.print_info()

# 使用 dev_ops 的 print_info() 方法
dev_ops.print_info()
```

![day1108](https://storage.googleapis.com/2017_ithome_ironman/day1108.png)

### 繼承（Inheritance）

新定義的類別可以繼承既有定義好的類別，可以沿用既有類別中的屬性及方法。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants
    
    def print_info(self):
        print(self.group, "組有", self.participants, "位鐵人參賽！")

# Articles 類別繼承 Ironmen 類別
class Articles(Ironmen):
    '''
    這是一個叫做 Articles 的類別。
    Articles 繼承 Ironmen 類別，她新增了一個 print_articles() 方法
    '''
    def print_articles(self):
        print(self.group, "組預計會有", self.participants * 30, "篇文章！")

# 根據 Articles 類別建立一個物件 modern_web
modern_web = Articles("Modern Web", 54)

# 使用 modern_web 的 print_articles() 方法
modern_web.print_articles()

# 檢查 modern_web 是否還擁有 print_info() 方法
modern_web.print_info()
```

![day1109](https://storage.googleapis.com/2017_ithome_ironman/day1109.png)

#### 在繼承時使用 super()

可以根據原本的屬性或方法之上建立新的屬性或方法。

```{python}
class OnlyGroup:
    '''這是一個叫做 OnlyGroup 的類別''' # Doc string
    def __init__(self, group):
        self.group = group

# Ironmen 類別繼承 OnlyGroup 類別
class Ironmen(OnlyGroup):
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        super().__init__(group)
        self.participants = participants

# 根據 Ironmen 類別建立一個物件 modern_web
modern_web = Ironmen("Modern Web", 54)

# 印出 modern_web 的兩個屬性
print(modern_web.group)
print(modern_web.participants)
```

![day1110](https://storage.googleapis.com/2017_ithome_ironman/day1110.png)

#### 在繼承時改寫方法（Override）

我們在先前繼承時成功增加一個方法 `print_articles()`，現在我們要試著在 Article 類別中改寫原本 Ironmen 類別中的 `print_info()` 方法。

```{python}
class Ironmen:
    '''這是一個叫做 Ironmen 的類別''' # Doc string
    def __init__(self, group, participants):
        self.group = group
        self.participants = participants
    
    def print_info(self):
        print(self.group, "組有", self.participants, "位鐵人參賽！")

# Articles 類別繼承 Ironmen 類別
class Articles(Ironmen):
    '''
    這是一個叫做 Articles 的類別。
    Articles 繼承 Ironmen 類別，她新增了一個 print_articles() 方法
    '''
    def print_articles(self):
        print(self.group, "組預計會有", self.participants * 30, "篇文章！")
    
    # 改寫 print_info() 方法
    def print_info(self):
        print(self.group, "組有", self.participants, "位鐵人參賽！p.s.我被改寫了！")

# 根據 Articles 類別建立一個物件 modern_web
modern_web = Articles("Modern Web", 54)

# 檢查 modern_web 的 print_info() 方法是否被改寫
modern_web.print_info()
```

![day1111](https://storage.googleapis.com/2017_ithome_ironman/day1111.png)

## 小結

第十一天我們討論 Python 的物件導向，我們透過簡單的範例來定義類別，在定義類別的時候指定屬於該類別的屬性與方法，然後建立出屬於該類別的物件，除此之外我們還討論了新增類別，新增方法與改寫方法。

## 參考連結

- [Introducing Python](http://shop.oreilly.com/product/0636920028659.do)
- [PROGRAMIZ](https://www.programiz.com/)