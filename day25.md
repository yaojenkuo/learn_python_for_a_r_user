# [第 25 天] 機器學習（5）整體學習

---

我們今天仍然繼續練習 Python 的 **scikit-learn** 機器學習套件，還記得在 [[第 23 天] 機器學習（3）決策樹與 k-NN 分類器](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day23.md)中我們建立了決策樹與 k-Nearest Neighbors 分類器嗎？當我們使用一種分類器沒有辦法達到很良好的預測結果時，除了改使用其他類型的分類器，還有一個方式稱為**整體學習（Ensemble learning）**可以將數個分類器的預測結果綜合考慮，藉此達到顯著提升分類效果。

那麼整體學習的概念是什麼？以做一題是非題來說，假如我們使用一個銅板來決定答案要填是還是非，答對的機率是 50%，如果使用兩個銅板來決定答案，答對的機率是 1-(50%\*50%)=75%，如果銅板的數目來到 5 枚，答對的機率是 1-(50%)^5=96.875%。隨著銅板的個數增加，答對這一題是非題的機率也隨之增加，這大概就是整體學習的基本理念。

> The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.
> [1.11. Ensemble methods - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/ensemble.html)

以前述例子而言，銅板就是所謂的基本分類器（Base estimator），或稱為弱分類器（Weak classifier），基本分類器的選擇是任意的，在經典的整體學習演算法 **Bagging** 與 **AdaBoost** 中我們多數使用決策樹作為基本分類器。跟 [[第 22 天] 機器學習（2）複迴歸與 Logistic 迴歸](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day22.md)練習 Logistic 迴歸一樣，我們繼續使用鐵達尼克號資料，分別在 Python 與 R 語言實作。

## Bagging

Bagging 是 Bootstrap Aggregating 的簡稱，透過統計學的 Bootstrap sampling 得到不同的訓練資料，然後根據這些訓練資料得到一系列的基本分類器，假如演算法產生了 5 個基本分類器，她們對某個觀測值的預測結果分別為 1, 0, 1, 1, 1，那麼 Bagging 演算法的輸出結果就會是 1，這個過程稱之為基本分類器的投票。

### Python

我們使用 `sklearn.ensemble` 的 `BaggingClassifier()`。

```python
import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics

# 載入資料
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 填補遺漏值
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立訓練與測試資料
titanic_X = pd.DataFrame([titanic_train["Pclass"],
                         encoded_Sex,
                         titanic_train["Age"]
]).T
titanic_y = titanic_train["Survived"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 bagging 模型
bag = ensemble.BaggingClassifier(n_estimators = 100)
bag_fit = bag.fit(train_X, train_y)

# 預測
test_y_predicted = bag.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
```

![day2501](https://storage.googleapis.com/2017_ithome_ironman/day2501.png)

### R 語言

我們使用 `adabag` 套件的 `bagging()` 函數。

```
library(adabag)
library(rpart)

url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train <- read.csv(url)
titanic_train$Survived <- factor(titanic_train$Survived)

# 將 Age 遺漏值以 median 填補
age_median <- median(titanic_train$Age, na.rm = TRUE)
new_Age <- ifelse(is.na(titanic_train$Age), age_median, titanic_train$Age)
titanic_train$Age <- new_Age

# 切分訓練與測試資料
n <- nrow(titanic_train)
shuffled_titanic <- titanic_train[sample(n), ]
train_indices <- 1:round(0.7 * n)
train_titanic <- shuffled_titanic[train_indices, ]
test_indices <- (round(0.7 * n) + 1):n
test_titanic <- shuffled_titanic[test_indices, ]

# 建立模型
bag_fit <- bagging(Survived ~ Pclass + Age + Sex, data = train_titanic, mfinal = 100)

# 預測
test_titanic_predicted <- predict(bag_fit, test_titanic)

# 績效
accuracy <- 1 - test_titanic_predicted$error
accuracy
```

![day2502](https://storage.googleapis.com/2017_ithome_ironman/day2502.png)

## AdaBoost

AdaBoost 同樣是基於數個基本分類器的整體學習演算法，跟前述 Bagging 演算法不同的地方在於，她在形成基本分類器時除了隨機生成，還會針對在前一個基本分類器中被分類錯誤的觀測值提高抽樣權重，使得該觀測值在下一個基本分類器形成時有更高機率被選入，藉此提高被正確分類的機率，簡單來說，她是個具有即時調節觀測值抽樣權重的進階 Bagging 演算法。

### Python

我們使用 `sklearn.ensemble` 的 `AdaBoostClassifier()`。

```python
import numpy as np
import pandas as pd
from sklearn import cross_validation, ensemble, preprocessing, metrics

# 載入資料
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 填補遺漏值
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立訓練與測試資料
titanic_X = pd.DataFrame([titanic_train["Pclass"],
                         encoded_Sex,
                         titanic_train["Age"]
]).T
titanic_y = titanic_train["Survived"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 boosting 模型
boost = ensemble.AdaBoostClassifier(n_estimators = 100)
boost_fit = boost.fit(train_X, train_y)

# 預測
test_y_predicted = boost.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
```

![day2503](https://storage.googleapis.com/2017_ithome_ironman/day2503.png)

### R 語言

我們使用 `adabag` 套件的 `boosting()` 函數。

```
library(adabag)
library(rpart)

url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train <- read.csv(url)
titanic_train$Survived <- factor(titanic_train$Survived)

# 將 Age 遺漏值以 median 填補
age_median <- median(titanic_train$Age, na.rm = TRUE)
new_Age <- ifelse(is.na(titanic_train$Age), age_median, titanic_train$Age)
titanic_train$Age <- new_Age

# 切分訓練與測試資料
n <- nrow(titanic_train)
shuffled_titanic <- titanic_train[sample(n), ]
train_indices <- 1:round(0.7 * n)
train_titanic <- shuffled_titanic[train_indices, ]
test_indices <- (round(0.7 * n) + 1):n
test_titanic <- shuffled_titanic[test_indices, ]

# 建立模型
boost_fit <- boosting(Survived ~ Pclass + Age + Sex, data = train_titanic, mfinal = 100)

# 預測
test_titanic_predicted <- predict(bag_fit, test_titanic)

# 績效
accuracy <- 1 - test_titanic_predicted$error
accuracy
```

![day2504](https://storage.googleapis.com/2017_ithome_ironman/day2504.png)

## 小結

第二十五天我們繼續練習 Python 的機器學習套件 **scikit-learn**，使用熟悉的鐵達尼克號資料，建立 Bagging 與 AdaBoost 的**整體學習**分類模型，並且也與 R 語言相互對照。

## 參考連結

- [1.11. Ensemble methods - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/ensemble.html)
- [Package ‘adabag’](https://cran.r-project.org/web/packages/adabag/adabag.pdf)