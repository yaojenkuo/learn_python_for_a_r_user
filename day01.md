# [第 01 天] 建立開發環境與計算機應用

---

從事資料科學相關工作的人，免不了在起步時都會思索：「假如時間有限，我應該選擇學習 R 語言或者 Python？」網路上相關的討論串已經太多，既然前提是「時間有限」，那我們更不應該花費時間去閱讀這些討論串，閱讀下來對於程式語言鄙視鏈的收穫可能還比原本的題目來得大。

這系列文章的視角是一個 R 語言使用者去學習 Python 資料科學的應用，希望讓還沒有開始學習的人對這兩個程式語言有一點 prior knowledge，藉由閱讀這系列文章，跟她們都稍微相處一下看看氛圍如何，再決定要選哪一個作為切入資料科學應用的程式語言。

## 學習筆記的脈絡

這份學習筆記從一個 R 語言使用者學習 Python 在資料科學的應用，並且相互對照的角度出發，整份學習筆記可以分為五大主題：

### 基礎

- [[第 01 天] 建立開發環境與計算機應用](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day01.md)
- [[第 02 天] 基本變數類型](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day02.md)
- [[第 03 天] 變數類型的轉換](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day03.md)
- [[第 04 天] 資料結構 List，Tuple 與 Dictionary](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day04.md)
- [[第 05 天] 資料結構（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day05.md)
- [[第 06 天] 資料結構（3）Data Frame](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day06.md)
- [[第 07 天] 迴圈與流程控制](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day07.md)
- [[第 08 天] 函數](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day08.md)
- [[第 09 天] 函數（2）](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day09.md)
- [[第 10 天] 物件導向 R 語言](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day10.md)
- [[第 11 天] 物件導向（2）Python](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day11.md)

### 基礎應用

- [[第 12 天] 常用屬性或方法 變數與基本資料結構](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day12.md)
- [[第 13 天] 常用屬性或方法（2）ndarray](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day13.md)
- [[第 14 天] 常用屬性或方法（3）Data Frame](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day14.md)
- [[第 15 天] 載入資料](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day15.md)
- [[第 16 天] 網頁解析](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day16.md)
- [[第 17 天] 資料角力](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day17.md)

### 視覺化

- [[第 18 天] 資料視覺化 matplotlib](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day18.md)
- [[第 19 天] 資料視覺化（2）Seaborn](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day19.md)
- [[第 20 天] 資料視覺化（3）Bokeh](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day20.md) 

### 機器學習

- [[第 21 天] 機器學習 玩具資料與線性迴歸](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day21.md)
- [[第 22 天] 機器學習（2）複迴歸與 Logistic 迴歸](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day22.md)
- [[第 23 天] 機器學習（3）決策樹與 k-NN 分類器](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day23.md)
- [[第 24 天] 機器學習（4）分群演算法](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day24.md)
- [[第 25 天] 機器學習（5）整體學習](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day25.md)
- [[第 26 天] 機器學習（6）隨機森林與支持向量機](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day26.md)

### 深度學習

- [[第 27 天] 深度學習 TensorFlow](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day27.md)
- [[第 28 天] 深度學習（2）TensorBoard](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day28.md)
- [[第 29 天] 深度學習（3）MNIST 手寫數字辨識](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day29.md)
- [[第 30 天] 深度學習（4）卷積神經網絡與鐵人賽總結](https://github.com/yaojenkuo/learn_python_for_a_r_user/blob/master/day30.md)

## 建立開發環境

R 語言的使用者可以在 [CRAN](https://cran.r-project.org/) 下載 R，然後再去 [RStudio](https://www.rstudio.com/products/rstudio/download3/) 下載這個好用的 IDE，兩者安裝完畢後就建立好了 R 語言的開發環境。

那麼關於 Python 的開發環境呢？我使用的作業系統是 OS X，系統已經安裝好 Python，只要打開終端機（不曉得怎麼開啟終端機的 Mac 使用者只要按 `Command + Space` 打開 Spotlight Search，搜尋 Terminal 然後按 `Enter`）輸入 `$ python` 就可以開始使用，如果需要編寫 .py 檔再執行的話我也可以使用慣用的 Sublime Text 或者任意的文字編輯器，感覺好像不需要再額外準備什麼東西。但是跟 RStudio 相較之下這樣的開發環境是略顯單薄了些，我覺得最基本起碼要能夠讓撰寫 .py 的編輯區跟命令列並陳，這樣開發起來才會舒服。

為了不要在第一天就歪腰，經過短暫的 google 之後，我打算使用 Jupyter Notebook 來作為我的 Python 開發環境。

### 安裝 Anaconda

Jupyter 官網的安裝建議沒有經驗的 Python 使用者透過 Anaconda 來安裝。Anaconda 是森蚺，南美洲的無毒蛇，跟蟒蛇（Python）都是體型非常巨大的蛇類，私心相當喜歡這個命名。

前往 [Anaconda](https://www.continuum.io/downloads) 將 .pkg 檔下載回來進行安裝。
OS X 原本安裝好的 Python 版本是 2.7，Jupyter 官網推薦安裝 Python 3 以上的版本，所以我選擇了 Python 3.5 版本的 Anaconda 4.2.0，安裝。Anaconda 安裝完畢後，在終端機輸入 `$ python` 確認安裝完成。

Python 3.5.2 |Anaconda 4.2.0 (x86_64)

輸入 `Ctrl + D` 離開 Python。

### 啟動 Jupyter Notebook

安裝 Anaconda 的同時也已經安裝 Jupyter Notebook，接著在終端機輸入以下指令啟動 Jupyter Notebook。

```
$ jupyter notebook
```

我們可以清楚得看到 Jupyter Notebook 是在 localhost:8888 上面運行，但是當我想要新增一個 Notebook 的時候，它出現的是 python [conda root] 與 python [default]。

![day0101](https://storage.googleapis.com/2017_ithome_ironman/day0101.png)

### 修正 Kernel 顯示問題

回到終端機按 `Ctrl + C` 停止 Jupyter Notebook，接著在終端機輸入指令。

```
$ conda remove _nb_ext_conf
```

重新啟動 Jupyter Notebook，在終端機輸入指令。

```
$ jupyter notebook
```

新增一個 Python 3 Notebook，問題順利排解。

![day0102](https://storage.googleapis.com/2017_ithome_ironman/day0102.png)

開發環境已經建立妥當了，接著讓我們在上面做最簡單的計算機應用吧！

## 計算機應用

在剛剛新增的 Python 3 Notebook 的第一個 cell 輸入一些簡單的加減乘除。

```python
print(2 + 3)
print(2 - 3)
print(2 * 3)
print(10 / 2)
print(3 ** 2) # R 語言使用 3 ^ 2
print(10 % 4) # R 語言使用 10 %% 4
```

輸入完後，選擇這個 cell 並在上方的工具列點選「Cell」後點選「Run Cells」，就會得到答案輸出。

![day0103](https://storage.googleapis.com/2017_ithome_ironman/day0103.png)

![day0105](https://storage.googleapis.com/2017_ithome_ironman/day0105.png)

跟 R 語言的運算子略有出入的地方在指數與餘數計算的部分。Python 使用 `**` 而非 `^` 來計算指數，使用 `%` 而非 `%%` 作餘數的計算。

## 小結

第一天我們介紹了怎麼在自己的電腦建立 Python 的開發環境，在上面做了簡單的計算機應用。在建立開發環境與計算機應用時也跟 R 語言比較了一下。

## 參考連結

- [Installing Jupyter](http://jupyter.org/install.html)
- [Download Anaconda Now!](https://www.continuum.io/downloads)
- [Issue #1716 jupyter/notebook](https://github.com/jupyter/notebook/issues/1716)