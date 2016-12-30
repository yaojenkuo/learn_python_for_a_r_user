# [第 22 天] 機器學習（2）複迴歸與 Logistic 迴歸

---

我們今天要繼續使用 **scikit-learn** 機器學習套件延續[昨天](http://ithelp.ithome.com.tw/articles/10186905)的線性迴歸，練習一個複迴歸以及一個 Logistic 迴歸。如果你還記得 [scikit-learn 首頁](http://scikit-learn.org/stable/index.html) 的應用領域，很明顯線性迴歸與複迴歸是屬於**迴歸（Regression）**應用領域，但是 Logistic 迴歸呢？她好像應當被歸類在**分類（Classification）**應用領域，但名字中又有迴歸兩個字？從 [Generalized Linear Models - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) 我們瞭解 Logistic 迴歸是預測機率的方法，屬於二元分類的應用領域。

> Logistic regression, despite its name, is a linear model for classification rather than regression.
> [Generalized Linear Models - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

接下來我們使用[世界第一簡單統計學迴歸分析篇](http://www.books.com.tw/products/0010479438)的蛋糕店數據來練習複迴歸，使用 [Kaggle](https://www.kaggle.com/) 著名的[鐵達尼克號資料](https://www.kaggle.com/c/titanic/data)來練習 Logistic 迴歸，並且分別在 Python 與 R 語言實作練習。

## 建立複迴歸模型

使用連鎖蛋糕店的 **店面面積（坪）**與**車站距離（公里）**來預測**分店單月銷售量（萬日圓）**。

### Python

我們使用 `sklearn.linear_model` 的 `LinearRegression()` 方法。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])

lm = LinearRegression()
lm.fit(X, y)

# 印出係數
print(lm.coef_)

# 印出截距
print(lm.intercept_ )
```

![day2201](https://storage.googleapis.com/2017_ithome_ironman/day2201.png)

### R 語言

我們使用 `lm()` 函數。

```
store_area <- c(10, 8, 8, 5, 7, 8, 7, 9, 6, 9)
dist_to_station <- c(80, 0, 200, 200, 300, 230, 40, 0, 330, 180)
monthly_sales <- c(469, 366, 371, 208, 246, 297, 363, 436, 198, 364)
bakery_df <- data.frame(store_area, dist_to_station, monthly_sales)

lm_fit <- lm(monthly_sales ~ ., data = bakery_df)

# 印出係數
lm_fit$coefficients[-1]

# 印出截距
lm_fit$coefficients[1]
```

![day2202](https://storage.googleapis.com/2017_ithome_ironman/day2202.png)

## 利用複迴歸模型預測

建立複迴歸模型之後，身為連鎖蛋糕店的老闆，在開設新店選址的時候，就可以用新店資訊預測單月銷售量，進而更精準地掌握店租與人事成本的管理。

### Python

我們使用 `LinearRegression()` 的 `predict()` 方法。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])

lm = LinearRegression()
lm.fit(X, y)

# 新蛋糕店資料
to_be_predicted = np.array([
    [10, 110]
])
predicted_sales = lm.predict(to_be_predicted)

# 預測新蛋糕店的單月銷量
print(predicted_sales)
```

![day2203](https://storage.googleapis.com/2017_ithome_ironman/day2203.png)

### R 語言

我們使用 `predict()` 函數。

```
store_area <- c(10, 8, 8, 5, 7, 8, 7, 9, 6, 9)
dist_to_station <- c(80, 0, 200, 200, 300, 230, 40, 0, 330, 180)
monthly_sales <- c(469, 366, 371, 208, 246, 297, 363, 436, 198, 364)
bakery_df <- data.frame(store_area, dist_to_station, monthly_sales)

lm_fit <- lm(monthly_sales ~ ., data = bakery_df)

# 新蛋糕店資料
to_be_predicted <- data.frame(store_area = 10, dist_to_station = 110)
predicted_sales <- predict(lm_fit, newdata = to_be_predicted)

# 預測新蛋糕店的單月銷量
predicted_sales
```

![day2204](https://storage.googleapis.com/2017_ithome_ironman/day2204.png)

## 複迴歸模型的績效

複迴歸模型的績效（Performance）有 **Mean squared error（MSE）**、 **R-squared** 與 **Adjusted R-squared**。

### Python

我們使用 `LinearRegression()` 方法建立出來物件的 `score` 屬性。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])

lm = LinearRegression()
lm.fit(X, y)

# 模型績效
mse = np.mean((lm.predict(X) - y) ** 2)
r_squared = lm.score(X, y)
adj_r_squared = r_squared - (1 - r_squared) * (X.shape[1] / (X.shape[0] - X.shape[1] - 1))

# 印出模型績效
print(mse)
print(r_squared)
print(adj_r_squared)
```

![day2205](https://storage.googleapis.com/2017_ithome_ironman/day2205.png)

### R 語言

使用 `summary(lm_fit)` 的 `r.squared` 與 `adj.r.squared` 屬性。

```
store_area <- c(10, 8, 8, 5, 7, 8, 7, 9, 6, 9)
dist_to_station <- c(80, 0, 200, 200, 300, 230, 40, 0, 330, 180)
monthly_sales <- c(469, 366, 371, 208, 246, 297, 363, 436, 198, 364)
bakery_df <- data.frame(store_area, dist_to_station, monthly_sales)

lm_fit <- lm(monthly_sales ~ ., data = bakery_df)
predicted_sales <- predict(lm_fit, newdata = data.frame(store_area, dist_to_station))

# 模型績效
mse <- mean((monthly_sales - predicted_sales) ^ 2)

# 印出模型績效
mse
summary(lm_fit)$r.squared
summary(lm_fit)$adj.r.squared
```

![day2206](https://storage.googleapis.com/2017_ithome_ironman/day2206.png)

## 複迴歸模型的係數檢定

複迴歸模型我們通常還會檢定變數的顯著性，以 **P-value** 是否小於 0.05（信心水準 95%）來判定。

### Python

我們使用 `sklearn.feature_selection` 的 `f_regression()` 方法。

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

X = np.array([
    [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180]
])
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364])

lm = LinearRegression()
lm.fit(X, y)

# 印出 p-value
print(f_regression(X, y)[1])
```

![day2207](https://storage.googleapis.com/2017_ithome_ironman/day2207.png)

### R 語言

使用 `summary(lm_fit)` 的 `coefficients` 屬性。

```
store_area <- c(10, 8, 8, 5, 7, 8, 7, 9, 6, 9)
dist_to_station <- c(80, 0, 200, 200, 300, 230, 40, 0, 330, 180)
monthly_sales <- c(469, 366, 371, 208, 246, 297, 363, 436, 198, 364)
bakery_df <- data.frame(store_area, dist_to_station, monthly_sales)

lm_fit <- lm(monthly_sales ~ ., data = bakery_df)

# 印出 p-value
summary(lm_fit)$coefficients[-1, 4]
```

![day2208](https://storage.googleapis.com/2017_ithome_ironman/day2208.png)

## 建立 Logistic 迴歸模型

在[Kaggle](https://www.kaggle.com/) 著名的鐵達尼克號資料，我們使用 **Sex**，**Pclass** 與 **Age** 來預測 **Survived**。

### Python

我們使用 `linear_model` 的 `LogisticRegression()` 方法。

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 將 Age 遺漏值以 median 填補
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age
titanic_train

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立 train_X
train_X = pd.DataFrame([titanic_train["Pclass"],
                        encoded_Sex,
                        titanic_train["Age"]
]).T

# 建立模型
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(train_X, titanic_train["Survived"])

# 印出係數
print(logistic_regr.coef_)

# 印出截距
print(logistic_regr.intercept_ )
```

![day2209](https://storage.googleapis.com/2017_ithome_ironman/day2209.png)

### R 語言

我們使用 `glm()` 函數，並指定參數 `family = binomial(link = "logit")`。

```
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train <- read.csv(url)

# 將 Age 遺漏值以 median 填補
age_median <- median(titanic_train$Age, na.rm = TRUE)
new_Age <- ifelse(is.na(titanic_train$Age), age_median, titanic_train$Age)
titanic_train$Age <- new_Age

# 建立模型
logistic_regr <- glm(Survived ~ Age + Pclass + Sex, data = titanic_train, family = binomial(link = "logit"))

# 印出係數
logistic_regr$coefficients[-1]

# 印出截距
logistic_regr$coefficients[1]
```

![day2210](https://storage.googleapis.com/2017_ithome_ironman/day2210.png)

## Logistic 迴歸模型係數檢定

Logistic 迴歸模型我們也可以檢定變數的顯著性，以 **P-value** 是否小於 0.05（信心水準 95%）來判定。

### Python

我們使用 `sklearn.feature_selection` 的 `f_regression()` 方法。

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import f_regression

url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 將 Age 遺漏值以 median 填補
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age
titanic_train

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立 train_X
train_X = pd.DataFrame([titanic_train["Pclass"],
                        encoded_Sex,
                        titanic_train["Age"]
]).T

# 建立模型
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(train_X, titanic_train["Survived"])

# 印出 p-value
print(f_regression(train_X, titanic_train["Survived"])[1])
```

![day2211](https://storage.googleapis.com/2017_ithome_ironman/day2211.png)

### R 語言

```
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train <- read.csv(url)

# 將 Age 遺漏值以 median 填補
age_median <- median(titanic_train$Age, na.rm = TRUE)
new_Age <- ifelse(is.na(titanic_train$Age), age_median, titanic_train$Age)
titanic_train$Age <- new_Age

# 建立模型
logistic_regr <- glm(Survived ~ Age + Pclass + Sex, data = titanic_train, family = binomial(link = "logit"))

# 印出 p-value
summary(logistic_regr)$coefficients[-1, 4]
```

![day2212](https://storage.googleapis.com/2017_ithome_ironman/day2212.png)

## Logistic 迴歸模型績效

我們用**準確率（Accuracy）**衡量二元分類模型的績效。

### Python

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 將 Age 遺漏值以 median 填補
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age
titanic_train

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立 train_X
train_X = pd.DataFrame([titanic_train["Pclass"],
                        encoded_Sex,
                        titanic_train["Age"]
]).T

# 建立模型
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(train_X, titanic_train["Survived"])

# 計算準確率
survived_predictions = logistic_regr.predict(train_X)
accuracy = logistic_regr.score(train_X, titanic_train["Survived"])
print(accuracy)
```

![day2213](https://storage.googleapis.com/2017_ithome_ironman/day2213.png)

### R 語言

```
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train <- read.csv(url)

# 將 Age 遺漏值以 median 填補
age_median <- median(titanic_train$Age, na.rm = TRUE)
new_Age <- ifelse(is.na(titanic_train$Age), age_median, titanic_train$Age)
titanic_train$Age <- new_Age

# 建立模型
logistic_regr <- glm(Survived ~ Age + Pclass + Sex, data = titanic_train, family = binomial(link = "logit"))

# 計算準確率
x_features <- titanic_train[, c("Age", "Pclass", "Sex")]
survived_predictions <- predict(logistic_regr, newdata = x_features, type = "response")
prediction_cutoff <- ifelse(survived_predictions > 0.5, 1, 0)
confusion_matrix <- table(titanic_train$Survived, prediction_cutoff)
accuracy <- sum(diag(confusion_matrix))/sum(confusion_matrix)
accuracy
```

![day2214](https://storage.googleapis.com/2017_ithome_ironman/day2214.png)

## 小結

第二十二天我們繼續練習使用 Python 的機器學習套件 **scikit-learn**，我們建立了一個複迴歸模型用店面面積與距車站距離預測蛋糕的單月銷售量；我們建立了一個 Logistic 迴歸模型用性別，年齡與社經地位預測鐵達尼號乘客的存活與否，並且與 R 語言相互對照。

## 參考連結

- [scikit-learn: machine learning in Python - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/index.html)
- [世界第一簡單統計學迴歸分析篇](http://www.books.com.tw/products/0010479438)
- [Python for Data Analysis Part 28: Logistic Regression](http://hamelg.blogspot.tw/2015/11/python-for-data-analysis-part-28.html)
- [Kaggle: Your Home for Data Science](https://www.kaggle.com/)