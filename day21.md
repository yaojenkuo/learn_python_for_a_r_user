# [第 21 天] 機器學習 玩具資料與線性迴歸

---

我們在 [[第 17 天] 資料角力](http://ithelp.ithome.com.tw/articles/10186310)提過，資料角力的目的是為了視覺化或者機器學習模型需求，必須將資料整理成合乎需求的格式。資料視覺化聽來直觀，那麼關於機器學習呢？我很喜歡[林軒田](http://www.csie.ntu.edu.tw/~htlin/)老師在[機器學習基石](https://www.youtube.com/watch?v=sS4523miLnw&list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf&index=2)簡單明嘹的解釋：

> 我們從小是怎麼樣辨認一棵樹的，是爸爸媽媽告訴我們一百條規則來定義嗎？其實不是的，很大一部分是透過我們自己的觀察很多的樹與不是樹之後，得到並且內化了辨認一棵樹的技巧，機器學習想要做的就是一樣的事情。
> [林軒田](http://www.csie.ntu.edu.tw/~htlin/)

我們要使用的 Python 機器學習套件是 **scikit-learn**，它建構於 **NumPy**、**SciPy** 與 **matplotlib** 之上，是開源套件並可作為商業使用。

> Scikit-learn is a Python module for machine learning built on top of SciPy and distributed under the 3-Clause BSD license. The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed. See the [AUTHORS.rst](https://github.com/scikit-learn/scikit-learn/blob/master/AUTHORS.rst) file for a complete list of contributors. It is currently maintained by a team of volunteers.
> [scikit-learn](https://github.com/scikit-learn/scikit-learn)

我們從 **scikit-learn** 套件的[首頁](http://scikit-learn.org/stable/index.html)可以一目瞭然它的應用領域：

- 監督式學習（Supervised learning）
    - 分類（Classification）
    - 迴歸（Regression）
- 非監督式學習（Unsupervised learning）
    - 分群（Clustering）
- 降維（Dimensionality reduction）
- 模型選擇（Model selection）
- 預處理（Preprocessing）

## 玩具資料（Toy datasets）

我們在練習資料視覺化或者機器學習的時候，除了可以自己產生資料以外，也可以用所謂的玩具資料（Toy datasets），玩具資料並不是一個特定的資料，而是泛指一些小而美的標準資料，像是在 R 語言中我們很習慣使用的 `iris`、`cars` 與 `mtcars` 資料框都是玩具資料。

### Python

我們使用 `sklearn` 的 `datasets` 物件的 `load_iris()` 方法來讀入鳶尾花資料。

```python
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
print(type(iris.data)) # 資料是儲存為 ndarray
print(iris.feature_names) # 變數名稱可以利用 feature_names 屬性取得
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names) # 轉換為 data frame
iris_df.ix[:, "species"] = iris.target # 將品種加入 data frame
iris_df.head() # 觀察前五個觀測值
```

![day2101](https://storage.googleapis.com/2017_ithome_ironman/day2101.png)

還有其他更多的玩具資料，像是波士頓房地產資料可以透過 `load_boston()` 方法讀入，糖尿病病患資料可以透過 `load_diabetes()` 方法讀入，詳情參考 [Dataset loading utilities - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/datasets/)。

### R 語言

`iris` 在 R 語言一啟動就已經讀入，可以直接使用。

```
head(iris)
```

![day2102](https://storage.googleapis.com/2017_ithome_ironman/day2102.png)

我們可以透過輸入 `data()` 得知有哪些資料可以直接使用。

![day210301](https://storage.googleapis.com/2017_ithome_ironman/day210301.png)

## 建立線性迴歸分析模型

我很喜歡[世界第一簡單統計學迴歸分析篇](http://www.books.com.tw/products/0010479438)的一個簡單例子：用氣溫來預測冰紅茶的銷售量。

### Python

我們使用 `sklearn.linear_model` 的 `LinearRegression()` 方法。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 印出係數
print(lm.coef_)

# 印出截距
print(lm.intercept_ )
```

![day2104](https://storage.googleapis.com/2017_ithome_ironman/day2104.png)

### R 語言

我們使用 `lm()` 函數。

```
temperatures <- c(29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30)
iced_tea_sales <- c(77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84)

lm_fit <- lm(iced_tea_sales ~ temperatures)

# 印出係數
lm_fit$coefficients[2]

# 印出截距
lm_fit$coefficients[1]
```

![day210501](https://storage.googleapis.com/2017_ithome_ironman/day210501.png)

## 利用線性迴歸分析模型預測

建立線性迴歸模型之後，身為冰紅茶店的老闆，就可以開始量測氣溫，藉此來預測冰紅茶銷量，更精準地掌握原料的管理。

### Python

我們使用 `LinearRegression()` 的 `predict()` 方法。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 新的氣溫
to_be_predicted = np.array([30])
predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))

# 預測的冰紅茶銷量
print(predicted_sales)
```

![day2106](https://storage.googleapis.com/2017_ithome_ironman/day2106.png)

### R 語言

我們使用 `predict()` 函數。

```
temperatures <- c(29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30)
iced_tea_sales <- c(77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84)

lm_fit <- lm(iced_tea_sales ~ temperatures)

# 新的氣溫
to_be_predicted <- data.frame(temperatures = 30)
predicted_sales <- predict(lm_fit, newdata = to_be_predicted)

# 預測的冰紅茶銷量
predicted_sales
```

![day2107](https://storage.googleapis.com/2017_ithome_ironman/day2107.png)

## 線性迴歸視覺化

我們可以使用 [[R 語言使用者的 Python 學習筆記 - 第 18 天] 資料視覺化](http://ithelp.ithome.com.tw/articles/10186484)提過的 Python `matplotlib` 套件與 R 語言的 **Base plotting system**。

### Python

我們使用 `matplotlib.pyplot` 的 `scatter()` 與 `plot()` 方法。

```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

lm = LinearRegression()
lm.fit(np.reshape(temperatures, (len(temperatures), 1)), np.reshape(iced_tea_sales, (len(iced_tea_sales), 1)))

# 新的氣溫
to_be_predicted = np.array([30])
predicted_sales = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))

# 視覺化
plt.scatter(temperatures, iced_tea_sales, color='black')
plt.plot(temperatures, lm.predict(np.reshape(temperatures, (len(temperatures), 1))), color='blue', linewidth=3)
plt.plot(to_be_predicted, predicted_sales, color = 'red', marker = '^', markersize = 10)
plt.xticks(())
plt.yticks(())
plt.show()
```

![day2108](https://storage.googleapis.com/2017_ithome_ironman/day2108.png)

### R 語言

我們使用 `plot()` 函數繪製散佈圖，使用 `points()` 繪製點，再使用 `abline()` 函數繪製直線。

```
temperatures <- c(29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30)
iced_tea_sales <- c(77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84)

lm_fit <- lm(iced_tea_sales ~ temperatures)

# 新的氣溫
to_be_predicted <- data.frame(temperatures = 30)
predicted_sales <- predict(lm_fit, newdata = to_be_predicted)

plot(iced_tea_sales ~ temperatures, bg = "blue", pch = 16)
points(x = to_be_predicted$temperatures, y = predicted_sales, col = "red", cex = 2, pch = 17)
abline(reg = lm_fit$coefficients, col = "blue", lwd = 4)
```

![day2109](https://storage.googleapis.com/2017_ithome_ironman/day2109.png)

## 線性迴歸模型的績效

線性迴歸模型的績效（Performance）有 **Mean squared error（MSE）**與 **R-squared**。

### Python

```python
import numpy as np
from sklearn.linear_model import LinearRegression

temperatures = np.array([29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30])
iced_tea_sales = np.array([77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84])

# 轉換維度
temperatures = np.reshape(temperatures, (len(temperatures), 1))
iced_tea_sales = np.reshape(iced_tea_sales, (len(iced_tea_sales), 1))

lm = LinearRegression()
lm.fit(temperatures, iced_tea_sales)

# 模型績效
mse = np.mean((lm.predict(temperatures) - iced_tea_sales) ** 2)
r_squared = lm.score(temperatures, iced_tea_sales)

# 印出模型績效
print(mse)
print(r_squared)
```

![day2110](https://storage.googleapis.com/2017_ithome_ironman/day2110.png)

### R 語言

```
temperatures <- c(29, 28, 34, 31, 25, 29, 32, 31, 24, 33, 25, 31, 26, 30)
iced_tea_sales <- c(77, 62, 93, 84, 59, 64, 80, 75, 58, 91, 51, 73, 65, 84)

lm_fit <- lm(iced_tea_sales ~ temperatures)
predicted_sales <- predict(lm_fit, newdata = data.frame(temperatures))

# 模型績效
mse <- mean((iced_tea_sales - predicted_sales) ^ 2)

# 印出模型績效
mse
summary(lm_fit)$r.squared
```

![day2111](https://storage.googleapis.com/2017_ithome_ironman/day2111.png)

## 小結

第二十一天我們練習使用 Python 的機器學習套件 **scikit-learn**，讀入一些玩具資料（Toy datasets），建立了一個簡單的線性迴歸模型來用氣溫預測冰紅茶銷量，並且與 R 語言的寫法相互對照。

## 參考連結

- [機器學習基石](https://www.youtube.com/watch?v=sS4523miLnw&list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf&index=2)
- [scikit-learn: machine learning in Python - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/index.html)
- [世界第一簡單統計學迴歸分析篇](http://www.books.com.tw/products/0010479438)