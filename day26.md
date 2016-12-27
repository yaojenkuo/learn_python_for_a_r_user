# [R 語言使用者的 Python 學習筆記 - 第 26 天] 機器學習（6）

---

我們今天繼續練習 Python 的 **scikit-learn** 機器學習套件，延續 [[R 語言使用者的 Python 學習筆記 - 第 25 天] 機器學習（5）](http://ithelp.ithome.com.tw/articles/10187452)的**整體學習（Ensemble learning）**，討論倍受歡迎的分類器**隨機森林（Random forest）**與**支持向量機（Support vector machine，SVM）**。

隨機森林分類器屬於建構於決策樹之上的整體學習應用，每一個基本分類器都是一個決策樹。這時我們心中就冒出一個疑問：隨機森林跟以決策樹為基本分類器構成的 Bagging 有什麼不同？最大的差異應該就是**隨機**的部分，以決策樹為基本分類器構成的 Bagging 的 Boostrap sampling 只有應用在列方向（觀測值方向）；隨機森林的 bootstrap sampling 則是同時應用在列方向（觀測值方向）與欄方向（變數方向）。

支持向量機則是一種利用最適化（Optimization）概念在模型的精確度以及推廣能力（Generalization ability）中取得一個最佳平衡點的演算法，她在面對小樣本，非線性與多維度的資料中廣受歡迎。

我們繼續使用鐵達尼克號資料，分別在 Python 與 R 語言實作兩種分類器。

## 隨機森林

隨機森林演算法會對資料從列方向（觀測值方向）與欄方向（變數方向）進行 Bootstrap sampling，得到不同的訓練資料，然後根據這些訓練資料得到一系列的決策樹分類器，假如產生了 5 個決策樹分類器，她們對某個觀測值的預測結果分別為 1, 0, 1, 1, 1，那麼隨機森林演算法的輸出結果就會是 1，這個過程與 Bagging 演算法相同，同樣稱為基本分類器的投票。隨機森林演算法在面對變數具有多元共線性或者不平衡資料（Unbalanced data）的情況時是倍受青睞的演算法。

> A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
> [3.2.4.3.1. sklearn.ensemble.RandomForestClassifier - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

### Python

我們使用 `sklearn.ensemble` 的 `RandomForestClassifier()`。

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

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)

# 預測
test_y_predicted = forest.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
```

![day2601](https://storage.googleapis.com/2017_ithome_ironman/day2601.png)

### R 語言

我們使用 `randomForest` 套件的 `randomForest()` 函數。

```
library(randomForest)

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
forest_fit <- randomForest(Survived ~ Pclass + Age + Sex, data = train_titanic, n_tree = 100)

# 預測
test_titanic_predicted <- predict(forest_fit, test_titanic)

# 績效
conf_matrix <- table(test_titanic_predicted, test_titanic$Survived)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy
```

![day2602](https://storage.googleapis.com/2017_ithome_ironman/day2602.png)

## 支持向量機

支持向量機是一種最小化結構風險（Structural risk）的演算法，何謂結構型風險？機器學習的內涵在於假設一個類似模型去逼近真實模型，而量化類似模型與真實模型之間差距的方式，跟我們在計算績效（準確率）用的概念是相同的，我們用類似模型預測的結果去跟答案比較。許多的分類器可以在訓練資料上達到很高的正確率（稱作 Overfitting），但是卻失去應用在實際問題的推廣能力（Generalization ability）。

資料科學家將分類器在訓練樣本可能過度配適的風險稱為 Empirical risk，分類器的推廣能力不足的風險稱為 Generalization risk，兩者的總和即為結構風險，而支持向量機就是在兩者之間取得最佳平衡點，進而得到一個在訓練資料績效不錯，亦能推廣適用的類似模型。

> Structural risk minimization (SRM) is an inductive principle for model selection used for learning from finite training data sets. It describes a general model of capacity control and provides a trade-off between hypothesis space complexity (the VC dimension of approximating functions) and the quality of fitting the training data (empirical error).
> <http://www.svms.org/>

### Python

我們使用 `sklearn.svm` 的 `SVC()`。

```python
import numpy as np
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics

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

# 建立 SVC 模型
svc = svm.SVC()
svc_fit = svc.fit(train_X, train_y)

# 預測
test_y_predicted = svc.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
```

![day2603](https://storage.googleapis.com/2017_ithome_ironman/day2603.png)

### R 語言

我們使用 `e1071` 套件的 `svm()` 函數。

```
library(e1071)

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
svm_fit <- svm(Survived ~ Pclass + Age + Sex, data = train_titanic)

# 預測
test_titanic_predicted <- predict(svm_fit, test_titanic)

# 績效
conf_matrix <- table(test_titanic_predicted, test_titanic$Survived)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy
```

![day2604](https://storage.googleapis.com/2017_ithome_ironman/day2604.png)

## AUC

在二元分類模型裡面，除了我們一直使用的準確率（Accuracy）以外，其實還有許多由 **Confusion matrix** 所衍生的績效指標，像是精確度（Precision）或者召回率（Recall）等，其中 **AUC** 是一個常見指標，它同時考慮假警報率（False alarm rate）與命中率（True positive rate），**AUC** 愈接近 1，就表示分類效果愈好；愈接近 0.5 就表示分類效果愈差不好。

| 效果 | AUC 區間 |
| --- | --- |
| 傑出 | AUC 介於 0.9-1.0 之間 |
| 優秀 | AUC 介於 0.8-0.9 之間 |
| 普通 | AUC 介於 0.7-0.8 之間 |
| 不好 | AUC 介於 0.6-0.7 之間 |
| 差勁 | AUC 介於 0.5-0.6 之間 |

我們來看看今天練習的隨機森林與支持向量機的 AUC 指標分別為何。

### 隨機森林分類器的 AUC

我們分別在 Python 使用 `sklearn.metrics` 的 `auc()`；在 R 語言使用 `ROCR` 套件的 `performance()` 函數。

#### Python

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

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)

# 預測
test_y_predicted = forest.predict(test_X)

# 績效
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
```

![day2605](https://storage.googleapis.com/2017_ithome_ironman/day2605.png)

我們建立的隨機森林模型分類效果為**優秀**。

#### R 語言

```
library(randomForest)
library(ROCR)

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
forest_fit <- randomForest(Survived ~ Pclass + Age + Sex, data = train_titanic)

# 預測
test_titanic_predicted_prob <- predict(forest_fit, test_titanic, type = "prob")
pred <- prediction(test_titanic_predicted_prob[, "1"], labels = test_titanic$Survived)

# 績效
perf <- performance(pred, "auc")
auc <- perf@y.values[[1]]
auc
```

![day2606](https://storage.googleapis.com/2017_ithome_ironman/day2606.png)

我們建立的隨機森林模型分類效果為**優秀**。

### 支持向量機分類器的 AUC

#### Python

```python
import numpy as np
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics

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

# 建立 SVC 模型
svc = svm.SVC()
svc_fit = svc.fit(train_X, train_y)

# 預測
test_y_predicted = svc.predict(test_X)

# 績效
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
```

![day2607](https://storage.googleapis.com/2017_ithome_ironman/day2607.png)

我們建立的支持向量機模型分類效果為**普通**。

#### R 語言

```
library(e1071)
library(ROCR)

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
svm_fit <- svm(Survived ~ Pclass + Age + Sex, data = train_titanic, probability = TRUE)

# 預測
test_titanic_predicted_prob <- predict(svm_fit, test_titanic, probability = TRUE)
pred <- prediction(attr(test_titanic_predicted_prob, "probabilities")[, "1"], labels = test_titanic$Survived)

# 績效
perf <- performance(pred, "auc")
auc <- perf@y.values[[1]]
auc
```

![day2608](https://storage.googleapis.com/2017_ithome_ironman/day2608.png)

我們建立的支持向量機模型分類效果為**優秀**。

## 小結

第二十六天我們繼續練習 Python 的機器學習套件 **scikit-learn**，使用熟悉的鐵達尼克號資料，使用隨機森林與支持向量機這兩個倍受歡迎的演算法建立了分類模型，在檢驗準確率之餘，我們今天也納入 AUC 作為模型績效評估指標，並且與 R 語言相互對照。

## 參考連結

- [3.2.4.3.1. sklearn.ensemble.RandomForestClassifier - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
- [sklearn.svm.SVC - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Package ‘randomForest’](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)
- [Package ‘e1071’](https://cran.r-project.org/web/packages/e1071/e1071.pdf)
- [Package ‘ROCR’](https://cran.r-project.org/web/packages/ROCR/ROCR.pdf)