# [第 24 天] 機器學習（4）分群演算法

---

我們今天依舊要繼續練習 **scikit-learn** 機器學習套件，經過三天的監督式學習（迴歸與分類）實作，稍微變換一下心情來練習非監督式學習中相當重要的分群演算法。仔細回想一下，至今練習過的冰紅茶銷量，蛋糕店月銷量，鐵達尼克號乘客的存活與否以及鳶尾花的品種，訓練資料都是**有標籤（答案）**的，而非監督式學習與監督式學習最大的不同之處就在於它的訓練資料是**沒有標籤（答案）**的。

分群演算法的績效衡量簡單明暸：**組間差異大，組內差異小**。而所謂的**差異**指的就是觀測值之間的距離遠近作為衡量，最常見還是使用[歐氏距離（Euclidean distance）](https://en.wikipedia.org/wiki/Euclidean_distance)。既然我們又是以距離作為度量，在資料的預處理程序中，與 k-Nearest Neighbors 分類器一樣我們必須將所有的數值型變數標準化（Normalization），避免因為單位不同，在距離的計算上失真。

我們今天要使用熟悉的鳶尾花資料，採用花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬來練習兩種分群演算法，分別是 **K-Means** 與 **Hierarchical Clustering**。

## K-Means

K-Means 演算法可以非常快速地完成分群任務，但是如果觀測值具有雜訊（Noise）或者極端值，其分群結果容易被這些雜訊與極端值影響，適合處理分布集中的大型樣本資料。

### 快速實作

#### Python

我們使用 `sklearn.cluster` 的 `KMeans()` 方法。

```python
from sklearn import cluster, datasets

# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(iris_X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
print("分群結果：")
print(cluster_labels)
print("---")

# 印出品種看看
iris_y = iris.target
print("真實品種：")
print(iris_y)
```

![day2401](https://storage.googleapis.com/2017_ithome_ironman/day2401.png)

看起來 setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異。

#### R 語言

我們使用 `kmeans()` 函數。

```
# 讀入鳶尾花資料
iris_kmeans <- iris[, -5]

# KMeans 演算法
kmeans_fit <- kmeans(iris_kmeans, nstart=20, centers=3)

# 印出分群結果
kmeans_fit$cluster

# 印出品種看看
iris$Species
```

![day2402](https://storage.googleapis.com/2017_ithome_ironman/day2402.png)

![day2403](https://storage.googleapis.com/2017_ithome_ironman/day2403.png)

看起來 setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異。

### 績效

分群演算法的績效可以使用 Silhouette 係數或 WSS（Within Cluster Sum of Squares）/BSS（Between Cluster Sum of Squares）。

#### Python

我們使用 `sklearn.metrics` 的 `silhouette_score()` 方法，這個數值愈接近 1 表示績效愈好，反之愈接近 -1 表示績效愈差。

> Compute the mean Silhouette Coefficient of all samples.
The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). The best value is 1 and the worst value is -1.
> [sklearn.metrics.silhouette_score - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)

```python
from sklearn import cluster, datasets, metrics

# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 3).fit(iris_X)
cluster_labels = kmeans_fit.labels_

# 印出績效
silhouette_avg = metrics.silhouette_score(iris_X, cluster_labels)
print(silhouette_avg)
```

![day2404](https://storage.googleapis.com/2017_ithome_ironman/day2404.png)

#### R 語言

我們使用 WSS（Within Cluster Sum of Squares）/BSS（Between Cluster Sum of Squares），這個數值愈低表示績效愈好。

```
# 讀入鳶尾花資料
iris_kmeans <- iris[, -5]

# KMeans 演算法
kmeans_fit <- kmeans(iris_kmeans, nstart=20, centers=3)
ratio <- kmeans_fit$tot.withinss / kmeans_fit$betweenss
ratio
```

![day240501](https://storage.googleapis.com/2017_ithome_ironman/day240501.png)

### 如何選擇 k

隨著 k 值的增加，K-Means 演算法的績效一定會愈來愈好，當 k = 觀測值數目的時候，我們會得到一個**組間差異最大，組內差異最小**的結果，但這不是我們想要的，實務上我們讓程式幫忙選擇一個適合的 k。

#### Python

```python
from sklearn import cluster, datasets, metrics
import matplotlib.pyplot as plt

# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data

# 迴圈
silhouette_avgs = []
ks = range(2, 11)
for k in ks:
    kmeans_fit = cluster.KMeans(n_clusters = k).fit(iris_X)
    cluster_labels = kmeans_fit.labels_
    silhouette_avg = metrics.silhouette_score(iris_X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

# 作圖並印出 k = 2 到 10 的績效
plt.bar(ks, silhouette_avgs)
plt.show()
print(silhouette_avgs)
```

![day2406](https://storage.googleapis.com/2017_ithome_ironman/day2406.png)

k 值在等於 2 與 3 的時候 K-Means 演算法的績效較好，這也驗證了我們先前的觀察，setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異，因此如果是以 K-Means 分群，可能會將 setosa 歸為一群，versicolor 和 virginica 歸為一群。

#### R 語言

```
# 讀入鳶尾花資料
iris_kmeans <- iris[, -5]

# 迴圈
ratio <- rep(NA, times = 10)
for (k in 2:length(ratio)) {
  kmeans_fit <- kmeans(iris_kmeans, centers = k, nstart = 20)
  ratio[k] <- kmeans_fit$tot.withinss / kmeans_fit$betweenss
}
plot(ratio, type="b", xlab="k")
```

![day240701](https://storage.googleapis.com/2017_ithome_ironman/day240701.png)

由上圖可以看出手肘點（Elbow point）出現在 k = 2 或者 k = 3 的時候，驗證了我們先前的觀察，setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異，因此如果是以 K-Means 分群，可能會將 setosa 歸為一群，versicolor 和 virginica 歸為一群。

## Hierarchical Clustering

與 K-Means 演算法不同的地方在於不需要事先設定 k 值，Hierarchical Clustering 演算法每一次只將兩個觀測值歸為一類，然後在演算過程中得到 k = 1 一直到 k = n（觀測值個數）群的結果。

### 快速實作

#### Python

```python
from sklearn import cluster, datasets

# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data

# Hierarchical Clustering 演算法
hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = 3)

# 印出分群結果
hclust.fit(iris_X)
cluster_labels = hclust.labels_
print(cluster_labels)
print("---")

# 印出品種看看
iris_y = iris.target
print(iris_y)
```

![day2408](https://storage.googleapis.com/2017_ithome_ironman/day2408.png)

看起來 setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異。

#### R 語言

```
# 讀入鳶尾花資料
iris_hclust <- iris[, -5]

# Hierarchical Clustering 演算法
dist_matrix <- dist(iris_hclust)
hclust_fit <- hclust(dist_matrix, method = "single")
hclust_fit_cut <- cutree(hclust_fit, k = 3)

# 印出分群結果
hclust_fit_cut

# 印出品種看看
iris$Species
```

![day2409](https://storage.googleapis.com/2017_ithome_ironman/day2409.png)

![day2410](https://storage.googleapis.com/2017_ithome_ironman/day2410.png)

看起來 setosa 這個品種跟另外兩個品種在花瓣（Petal）的長和寬跟花萼（Sepal）的長和寬有比較大的差異。

### 績效

#### Python

```python
from sklearn import cluster, datasets, metrics

# 讀入鳶尾花資料
iris = datasets.load_iris()
iris_X = iris.data

# Hierarchical Clustering 演算法
hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = 3)

# 印出績效
hclust.fit(iris_X)
cluster_labels = hclust.labels_
silhouette_avg = metrics.silhouette_score(iris_X, cluster_labels)
print(silhouette_avg)
```

![day2411](https://storage.googleapis.com/2017_ithome_ironman/day2411.png)

#### R 語言

```
library(GMD)

# 讀入鳶尾花資料
iris_hclust <- iris[, -5]

# Hierarchical Clustering 演算法
dist_matrix <- dist(iris_hclust)
hclust_fit <- hclust(dist_matrix)
hclust_fit_cut <- cutree(hclust_fit, k = 3)

# 印出績效
hc_stats <- css(dist_matrix, clusters = hclust_fit_cut)
hc_stats$totwss / hc_stats$totbss

# Dendrogram
plot(hclust_fit)
rect.hclust(hclust_fit, k = 2, border = "red")
rect.hclust(hclust_fit, k = 3, border = "green")
```

![day241201](https://storage.googleapis.com/2017_ithome_ironman/day241201.png)

![day2413](https://storage.googleapis.com/2017_ithome_ironman/day2413.png)

從 Dendrogram 看起來 setosa 這個品種跟兩個品種有比較大的差異，在 k = 2 或 k = 3 時都會被演算法歸類為獨立一群。

## 小結

第二十四天我們繼續練習 Python 的機器學習套件 **scikit-learn**，延續使用熟悉的鳶尾花資料，建立非監督式學習的 K-Means 與 Hierarchical Clustering 的分群模型，在分群演算法之下，我們發現 setosa 品種與 versicolor 及 virginica 的在花瓣與萼片的差異較大，而另兩品種則比較相近，並且也與 R 語言相互對照。

## 參考連結

- [scikit-learn: machine learning in Python - scikit-learn 0.18.1 documentation](http://scikit-learn.org/stable/index.html)
- [Package ‘GMD’](https://cran.r-project.org/web/packages/GMD/GMD.pdf)