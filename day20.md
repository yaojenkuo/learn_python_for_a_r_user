# [第 20 天] 資料視覺化（3）Bokeh

---

我們前兩天討論的 **matplotlib** 與 **Seaborn** 套件基本上已經可以滿足絕大多數的繪圖需求，唯一美中不足的一點是這些圖形都是靜態（Static）的，如果我們想要讓這些圖形帶有一點互動（Interactive），像是滑鼠游標移上去會顯示資料點的數據或可以縮放圖形等基於 JavaScript 的效果，我們可以在 Python 使用 **Bokeh** 這個高階繪圖套件來達成。

> Bokeh is a Python interactive visualization library that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of novel graphics in the style of D3.js, and to extend this capability with high-performance interactivity over very large or streaming datasets. Bokeh can help anyone who would like to quickly and easily create interactive plots, dashboards, and data applications.
> [Welcome to Bokeh - Bokeh 0.12.3 documentation](http://bokeh.pydata.org/en/latest/)

以目前發展來看，資料視覺化套件會以 [d3.js](https://d3js.org/) 與其他基於 JavaScript 或 [d3.js](https://d3js.org/) 衍生的網頁專案領銜衝鋒，像是 [Leaflet](http://leafletjs.com/)、[c3.js](http://c3js.org/) 以及我們今天會使用的 [plotly](https://plot.ly/feed/) 等。Python 與 R 語言的視覺化套件則是努力讓使用者用精簡的方法與函數畫出具備互動效果的視覺化，如我們今天要討論的 **Bokeh** 以及 **Plotly**。如果你的工作是以資料視覺化為重，花時間與精神鑽研網頁前端技術與 [d3.js](https://d3js.org/) 是必須的投資。

> Visualizations built on web technologies (that is, JavaScript-based) appear to be the inevitable future.
> [Wes McKinney](http://wesmckinney.com/)

我們今天試著使用 **Bokeh** 與 R 語言的 **Plotly** 套件來畫一些基本的圖形，包括：

- 直方圖（Histogram）
- 散佈圖（Scatter plot）
- 線圖（Line plot）
- 長條圖（Bar plot）
- 盒鬚圖（Box plot）

我下載的 [Anaconda](https://www.continuum.io/why-anaconda) 版本已經將 **Bokeh** 安裝好了，如果你的版本沒有，只要在終端機執行這段程式即可。

```
$ conda install -c anaconda bokeh=0.12.3
```

## 直方圖（Histogram）

### Python

使用 `bokeh.charts` 的 `Histogram()` 方法。

```python
from bokeh.charts import Histogram, show
import numpy as np

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
hist = Histogram(normal_samples)
show(hist)
```

![day2001](https://storage.googleapis.com/2017_ithome_ironman/day2001.png)

### R 語言

使用 `ggplotly()` 函數就可以將 **ggplot2** 套件所繪製的基本圖形轉換為 **Plotly** 圖形。

```
library(ggplot2)
library(plotly)

normal_samples <- rnorm(100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples_df <- data.frame(normal_samples)
hist <- ggplot(normal_samples_df, aes(x = normal_samples)) + geom_histogram(aes(y = ..density..)) + geom_density()
ggplotly(hist)
```

![day2002](https://storage.googleapis.com/2017_ithome_ironman/day2002.png)

## 散佈圖（Scatter plot）

### Python

使用 `bokeh.charts` 的 `Scatter()` 方法。

```python
from bokeh.charts import Scatter, show
import pandas as pd

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

scatter = Scatter(cars_df, x = "speed", y = "dist")
show(scatter)
```

![day2003](https://storage.googleapis.com/2017_ithome_ironman/day2003.png)

### R 語言

使用 `ggplotly()` 函數就可以將 **ggplot2** 套件所繪製的基本圖形轉換為 **Plotly** 圖形。

```
library(ggplot2)
library(plotly)

scatter_plot <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
ggplotly(scatter_plot)
```

![day2004](https://storage.googleapis.com/2017_ithome_ironman/day2004.png)

## 線圖（Line plot）

### Python

使用 `bokeh.charts` 的 `Line()` 方法。

```python
from bokeh.charts import Line, show
import pandas as pd

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

line = Line(cars_df, x = "speed", y = "dist")
show(line)
```

![day2005](https://storage.googleapis.com/2017_ithome_ironman/day2005.png)

### R 語言

使用 `ggplotly()` 函數就可以將 **ggplot2** 套件所繪製的基本圖形轉換為 **Plotly** 圖形。

```
library(ggplot2)
library(plotly)

line <- ggplot(cars, aes(x = speed, y = dist)) + geom_line()
ggplotly(line)
```

![day2006](https://storage.googleapis.com/2017_ithome_ironman/day2006.png)

## 長條圖（Bar plot）

### Python

使用 `bokeh.charts` 的 `Bar()` 方法。

```
from bokeh.charts import Bar, show
import pandas as pd

cyls = [11, 7, 14]
labels = ["4", "6", "8"]
cyl_df = pd.DataFrame({
    "cyl": cyls,
    "label": labels
})

bar = Bar(cyl_df, values = "cyl", label = "label")
show(bar)
```

![day2007](https://storage.googleapis.com/2017_ithome_ironman/day2007.png)

### R 語言

使用 `ggplotly()` 函數就可以將 **ggplot2** 套件所繪製的基本圖形轉換為 **Plotly** 圖形。

```
library(ggplot2)
library(plotly)

bar <- ggplot(mtcars, aes(x = cyl)) + geom_bar()
ggplotly(bar)
```

![day2008](https://storage.googleapis.com/2017_ithome_ironman/day2008.png)

## 盒鬚圖（Box plot）

### Python

使用 `bokeh.charts` 的 `BoxPlot()` 方法。

```python
from bokeh.charts import BoxPlot, show, output_notebook
import pandas as pd

output_notebook()

mpg = [21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26, 30.4, 15.8, 19.7, 15, 21.4]
cyl = [6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 4, 4, 4, 8, 6, 8, 4]
mtcars_df = pd.DataFrame({
    "mpg": mpg,
    "cyl": cyl
})

box = BoxPlot(mtcars_df, values = "mpg", label = "cyl")
show(box)
```

![day2009](https://storage.googleapis.com/2017_ithome_ironman/day2009.png)

### R 語言

使用 `ggplotly()` 函數就可以將 **ggplot2** 套件所繪製的基本圖形轉換為 **Plotly** 圖形。

```
library(ggplot2)
library(plotly)

box <- ggplot(mtcars, aes(y = mpg, x = factor(cyl))) + geom_boxplot()
ggplotly(box)
```

![day2010](https://storage.googleapis.com/2017_ithome_ironman/day2010.png)

## 小結

第二十天我們練習使用 Python 的視覺化套件 **Bokeh** 繪製基本的圖形，並且在 R 語言中使用 `plotly` 套件的 `ggplotly()` 函數將 **ggplot2** 的圖形轉換為互動的 **Plotly** 圖形。

## 參考連結

- [Making High-level Charts - Bokeh 0.12.3 documentation](http://bokeh.pydata.org/en/latest/docs/user_guide/charts.html)
- [ggplot2 | plotly](https://plot.ly/ggplot2/)
- [ggplot2 0.9.3.1](http://docs.ggplot2.org/0.9.3.1/index.html)
- [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do)