# [R 語言使用者的 Python 學習筆記 - 第 16 天] 網頁解析

---

並不是所有的資料都能這麼方便地以表格式資料（Tabular data），EXCEL 試算表或者 JSON 載入工作環境，有時候我們的資料散落在網路不同的角落裡，然而並不是每一個網站都會建置 API（Application Programming Interface）讓你很省力地把資料帶回家，這時候我們就會需要網頁解析（Web scraping）。

R 語言使用者對於 `rvest` 套件在網頁解析的便利性愛不釋手，而 Python 對應的代表套件就是 `BeautifulSoup`，事實上，`rvest` 開發的靈感就是啟發自 `BeautifulSoup`。
 
> rvest helps you scrape information from web pages. It is designed to work with magrittr to make it easy to express common web scraping tasks, inspired by libraries like beautiful soup.
> [Hadley Wickham](http://hadley.nz)

## 準備工作

除了 `BeautifulSoup` 套件以外，我們還需要搭配使用 `lxml` 套件與 `requests` 套件。由於我們的開發環境是安裝 [Anaconda]()，所以這些套件都不需要再另外下載與安裝，只要進行一貫的 `import` 就好。如果對於開發環境的部分有興趣，我推薦你參考 [[R 語言使用者的 Python 學習筆記 - 第 01 天] 建立開發環境與計算機應用](http://ithelp.ithome.com.tw/articles/10184561)。

### lxml 套件

`lxml` 套件是用來作為 `BeautifulSoup` 的解析器（Parser），`BeautifulSoup` 可以支援的解析器其實不只一種，還有 `html.parser`（Python 內建）與 `html5lib`，根據官方文件的推薦，我們使用解析速度最快的 `lxml`。

> If you can, I recommend you install and use lxml for speed.
> [Beautiful Soup Documentation — Beautiful Soup 4.4.0 documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc)

### requests 套件

`requests` 套件允許我們發送與接收**有機**及**草飼**的 HTTP/1.1 請求（這真的是美式幽默。）

> Requests allows you to send organic, grass-fed HTTP/1.1 requests, without the need for manual labor. There's no need to manually add query strings to your URLs, or to form-encode your POST data. Keep-alive and HTTP connection pooling are 100% automatic, powered by urllib3, which is embedded within Requests.
> [Requests: HTTP for Humans](http://docs.python-requests.org/en/master/)

## 第一個 BeautifulSoup 應用

先喝一口美麗的湯嚐嚐味道。

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器
print(soup.prettify()) # 把排版後的 html 印出來
```

![day1601](https://storage.googleapis.com/2017_ithome_ironman/day1601.png)

### 一些 BeautifulSoup 的屬性或方法

很快試用一些 BeautifulSoup 的屬性或方法。

- `title` 屬性
- `title.name` 屬性
- `title.string` 屬性
- `title.parent.name` 屬性
- `a` 屬性
- `find_all()` 方法

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

# 一些屬性或方法
print(soup.title) # 把 tag 抓出來
print("---")
print(soup.title.name) # 把 title 的 tag 名稱抓出來
print("---")
print(soup.title.string) # 把 title tag 的內容欻出來
print("---")
print(soup.title.parent.name) # title tag 的上一層 tag
print("---")
print(soup.a) # 把第一個 <a></a> 抓出來
print("---")
print(soup.find_all('a')) # 把所有的 <a></a> 抓出來
```

![day1602](https://storage.googleapis.com/2017_ithome_ironman/day1602.png)

## bs4 元素

`Beautiful Soup` 幫我們將 html 檔案轉換為 bs4 的物件，像是標籤（Tag），標籤中的內容（NavigableString）與 BeautifulSoup 物件本身。

### 標籤（Tag）

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(type(soup.a))
print("---")
print(soup.a.name) # 抓標籤名 a
print("---")
print(soup.a['id']) # 抓<a></a>的 id 名稱
```

![day1603](https://storage.googleapis.com/2017_ithome_ironman/day1603.png)

### 標籤中的內容（NavigableString）

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(type(soup.a.string))
print("---")
soup.a.string
```

![day1604](https://storage.googleapis.com/2017_ithome_ironman/day1604.png)

### BeautifulSoup

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, 'lxml') # 指定 lxml 作為解析器

type(soup)
```

![day1605](https://storage.googleapis.com/2017_ithome_ironman/day1605.png)

## 爬樹

DOM（Document Object Model）的樹狀結構觀念在使用 `BeautifulSoup` 扮演至關重要的角色，所以我們也要練習爬樹。

### 往下爬

從標籤中回傳更多資訊。

- `contents` 屬性
- `children` 屬性
- `string` 屬性

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(soup.body.a.contents)
print(list(soup.body.a.children))
print(soup.body.a.string)
```

![day1606](https://storage.googleapis.com/2017_ithome_ironman/day1606.png)

### 往上爬

回傳上一階層的標籤。

- `parent` 屬性

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(soup.title)
print("---")
print(soup.title.parent)
```

![day1607](https://storage.googleapis.com/2017_ithome_ironman/day1607.png)

### 往旁邊爬

回傳同一階層的標籤。

- `next_sibling` 屬性
- `previous_sibling` 屬性

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

first_a_tag = soup.body.a
next_to_first_a_tag = first_a_tag.next_sibling
print(first_a_tag)
print("---")
print(next_to_first_a_tag)
print("---")
print(next_to_first_a_tag.previous_sibling)
```

![day1608](https://storage.googleapis.com/2017_ithome_ironman/day1608.png)

## 搜尋

這是我們主要使用 `BeautifulSoup` 套件來做網站解析的方法。

- `find()` 方法
- `find_all()` 方法

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(soup.find("a")) # 第一個 <a></a>
print("---")
print(soup.find_all("a")) # 全部 <a></a>
```

![day1609](https://storage.googleapis.com/2017_ithome_ironman/day1609.png)

可以在第二個參數 `class_= ` 加入 CSS 的類別。

```python
import requests as rq
from bs4 import BeautifulSoup

url = "https://www.ptt.cc/bbs/NBA/index.html" # PTT NBA 板
response = rq.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

print(soup.find("div", class_= "r-ent"))
```

![day1610](https://storage.googleapis.com/2017_ithome_ironman/day1610.png)

## BeautifulSoup 牛刀小試

大略照著[官方文件](https://www.crummy.com/software/BeautifulSoup/bs4/doc)練習了前面的內容之後，我們參考[Tutorial of PTT crawler](https://github.com/leVirve/CrawlerTutorial)來應用 `BeautifulSoup` 把 PTT NBA 版首頁資訊包含推文數，作者 id，文章標題與發文日期搜集下來。

我們需要的資訊都放在 CSS 類別為 `r-ent` 的 `<div></div>` 中。

```python
import requests as rq
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/NBA/index.html'
response = rq.get(url)
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

posts = soup.find_all("div", class_ = "r-ent")
print(posts)
type(posts)
```

![day1611](https://storage.googleapis.com/2017_ithome_ironman/day1611.png)

注意這個 `posts` 物件是一個 `ResultSet`，一般我們使用迴圈將裡面的每一個元素再抓出來，先練習一下作者 id。

```python
import requests as rq
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/NBA/index.html'
response = rq.get(url)
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

author_ids = [] # 建立一個空的 list 來放置作者 id
posts = soup.find_all("div", class_ = "r-ent")
for post in posts:
    author_ids.extend(post.find("div", class_ = "author"))

print(author_ids)
```

![day1612](https://storage.googleapis.com/2017_ithome_ironman/day1612.png)

接下來我們把推文數，文章標題與發文日期一起寫進去。

```python
import numpy as np
import requests as rq
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/NBA/index.html'
response = rq.get(url)
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

author_ids = [] # 建立一個空的 list 來放作者 id
recommends = [] # 建立一個空的 list 來放推文數
post_titles = [] # 建立一個空的 list 來放文章標題
post_dates = [] # 建立一個空的 list 來放發文日期

posts = soup.find_all("div", class_ = "r-ent")
for post in posts:
    try:
        author_ids.append(post.find("div", class_ = "author").string)    
    except:
        author_ids.append(np.nan)
    try:
        post_titles.append(post.find("a").string)
    except:
        post_titles.append(np.nan)
    try:
        post_dates.append(post.find("div", class_ = "date").string)
    except:
        post_dates.append(np.nan)

# 推文數藏在 div 裡面的 span 所以分開處理
recommendations = soup.find_all("div", class_ = "nrec")
for recommendation in recommendations:
    try:
        recommends.append(int(recommendation.find("span").string))
    except:
        recommends.append(np.nan)

print(author_ids)
print(recommends)
print(post_titles)
print(post_dates)
```

![day1613](https://storage.googleapis.com/2017_ithome_ironman/day1613.png)

檢查結果都沒有問題之後，那我們就可以把這幾個 list 放進 dictionary 接著轉換成 data frame 了。

```python
import numpy as np
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/NBA/index.html'
response = rq.get(url)
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器

author_ids = [] # 建立一個空的 list 來放作者 id
recommends = [] # 建立一個空的 list 來放推文數
post_titles = [] # 建立一個空的 list 來放文章標題
post_dates = [] # 建立一個空的 list 來放發文日期

posts = soup.find_all("div", class_ = "r-ent")
for post in posts:
    try:
        author_ids.append(post.find("div", class_ = "author").string)    
    except:
        author_ids.append(np.nan)
    try:
        post_titles.append(post.find("a").string)
    except:
        post_titles.append(np.nan)
    try:
        post_dates.append(post.find("div", class_ = "date").string)
    except:
        post_dates.append(np.nan)

# 推文數藏在 div 裡面的 span 所以分開處理
recommendations = soup.find_all("div", class_ = "nrec")
for recommendation in recommendations:
    try:
        recommends.append(int(recommendation.find("span").string))
    except:
        recommends.append(np.nan)
        
ptt_nba_dict = {"author": author_ids,
                "recommends": recommends,
                "title": post_titles,
                "date": post_dates
}

ptt_nba_df = pd.DataFrame(ptt_nba_dict)
ptt_nba_df
```

![day1614](https://storage.googleapis.com/2017_ithome_ironman/day1614.png)

## rvest 牛刀小試

```
library(rvest)
library(magrittr)

ptt_nba_parser <- function() {
    url <- "https://www.ptt.cc/bbs/NBA/index.html"
    html_doc <- read_html(url)
    
    # 指定 xpath
    xpath_author_ids <- "//div[@class='meta']/div[@class='author']"
    xpath_recommends <- "//div[@class='nrec']"
    xpath_titles <- "//div[@class='title']"
    xpath_dates <- "//div[@class='meta']/div[@class='date']"
    
    # 擷取資料
    author_ids <- html_doc %>% 
        html_nodes(xpath = xpath_author_ids) %>%
        html_text    
    recommends <- html_doc %>%
        html_nodes(xpath = xpath_recommends) %>%
        html_text %>%
        as.integer
    titles <- html_doc %>%
        html_nodes(xpath = xpath_titles) %>%
        html_text
    dates <- html_doc %>%
        html_nodes(xpath = xpath_dates) %>%
        html_text
    
    # 整理成 data frame
    df <- data.frame(author_id = author_ids, recommends = recommends, title = titles, date = dates)
    return(df)
}

ptt_nba_df <- ptt_nba_parser()
View(ptt_nba_df)
```

![day1615](https://storage.googleapis.com/2017_ithome_ironman/day1615.png)

![day1616](https://storage.googleapis.com/2017_ithome_ironman/day1616.png)

## 關於牛刀小試的注意事項

- `BeautifulSoup` 我們使用的選擇概念是 CSS 選擇器；`rvest` 我們則是使用 XPATH 選擇器
- 兩種作法都需要考慮同一個基本問題，就是被刪除的文章，在 Python 中我們使用 `try-except` 讓程式不會中斷，在 R 語言中我們用更廣泛的方式指定 XPATH。

## 小結

第 16 天我們稍微練習了一下 Python 極富盛名的網頁解析套件 `BeautifulSoup` ，我們做了官方文件的一些範例以及 PTT 的練習，並也使用 R 語言的 `rvest` 套件做 PTT 的練習相互對照。

## 參考連結

- [Beautiful Soup Documentation - Beautiful Soup 4.4.0 documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [rvest@GitHub](https://github.com/hadley/rvest)
- [Tutorial of PTT crawler](https://github.com/leVirve/CrawlerTutorial)