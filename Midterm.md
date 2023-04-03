## Imports


```python
import fnmatch
import glob
import os
import re
from time import sleep
from zipfile import ZipFile
from requests_html import HTMLSession

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from near_regex import NEAR_regex  # copy this file into the asgn folder
from tqdm import tqdm

os.makedirs("output", exist_ok=True)
os.makedirs('inputs', exist_ok=True)
os.makedirs('10K_Files', exist_ok=True)
```

## Downlaod data

## S&P 500 Firms


```python
sp500_file = 'inputs/sp500_2022.csv'

if not os.path.exists(sp500_file):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    pd.read_html(url)[0].to_csv('inputs/sp500_2022.csv', index= False)
    
sp500 = pd.read_csv('inputs/sp500_2022.csv')

```

## Stock Returns Data


```python
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

fpath = 'inputs/Stock_Returns_CRSP/crsp_2022_only.zip'
sret = pd.read_stata(fpath)   
sret
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>date</th>
      <th>ret</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JJSF</td>
      <td>2021-12-01</td>
      <td>-0.011276</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JJSF</td>
      <td>2021-12-02</td>
      <td>0.030954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JJSF</td>
      <td>2021-12-03</td>
      <td>0.000287</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JJSF</td>
      <td>2021-12-06</td>
      <td>0.014362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JJSF</td>
      <td>2021-12-07</td>
      <td>0.012459</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2594044</th>
      <td>TSLA</td>
      <td>2022-12-23</td>
      <td>-0.017551</td>
    </tr>
    <tr>
      <th>2594045</th>
      <td>TSLA</td>
      <td>2022-12-27</td>
      <td>-0.114089</td>
    </tr>
    <tr>
      <th>2594046</th>
      <td>TSLA</td>
      <td>2022-12-28</td>
      <td>0.033089</td>
    </tr>
    <tr>
      <th>2594047</th>
      <td>TSLA</td>
      <td>2022-12-29</td>
      <td>0.080827</td>
    </tr>
    <tr>
      <th>2594048</th>
      <td>TSLA</td>
      <td>2022-12-30</td>
      <td>0.011164</td>
    </tr>
  </tbody>
</table>
<p>2594049 rows × 3 columns</p>
</div>



## Create Variables

### 2 versions of a “buy and hold” around the 10-K date (“date t”)

Version 1: Measure from the day t to day t+2 (inclusive)- t is bus days, so ignore weekend/holidays

Version 2: Measure from the day t+3 to day t+10 (inclusive)

### 10 Sentiment Variables

1. The first 4 of the 10 variables:
- “LM Positive” and “LM Negative”
- “ML Positive” and “ML Negative”


```python
#ML negative words into a list called BHR_negative

BHR_neg = pd.read_csv('inputs/ML_negative_unigram.txt',
            names=['word'])['word'].to_list()  
```


```python
regex_BHR_neg = ['('+'|'.join(BHR_neg)+')'] 
```


```python
#ML positive words into a list called BHR_positive

with open('inputs/ML_positive_unigram.txt', 'r') as file:
    #r means we're opening the file in "read" mode opposed to "write" mode
    BHR_pos = [line.strip() for line in file]
    
```


```python
regex_BHR_pos = ['('+'|'.join(BHR_pos)+')'] 
regex_BHR_pos
```




    ['(strong|strength|great|improvement|nice|improved|momentum|congratulations|pleased|helped|impressive|exceeded|record|congrats|good|leverage|raising|sustainable|really|job|benefited|continue|outperformance|increased|excellent|growth|increase|driving|helping|drove|grew|performance|pretty|above|margin|better|curious|across|continued|results|up|increasing|share|outstanding|improvements|operating|success|expansion|income|over|benefiting|lot|terrific|growing|favorable|generated|proud|repurchase|exceeding|solid|benefit|nicely|basis|flow|gains|well|achieved|upside|improving|cash|years|continues|delivered|think|fantastic)']




```python
# LM negative words into a list called LM_negative

LM = pd.read_csv('inputs/LM_MasterDictionary_1993-2021.csv')
LM_neg = LM.query('Negative == 2009')
```


```python
regex_LM_neg = ['('+'|'.join(LM_neg)+')'] 
```


```python
# load the LM postive words into a list called LM_positive

LM = pd.read_csv('inputs/LM_MasterDictionary_1993-2021.csv')
LM_pos = LM.query('Positive == 2009')
```


```python
regex_LM_pos = ['('+'|'.join(LM_pos)+')'] 
```

2. The last 6 of the 10 variables: “Contextual” sentiment. Basically: What is the (positive and negative) sentiment of the text in a 10-K around discussions of a particular topic.
- Pick three topics. Each of those will get a positive and negative sentiment score.
- https://ledatascifi.github.io/ledatascifi-2023/content/assignments/contextual_sentiment.html


```python
debt_list = ['(leverage|bonds|solvency|credit|d/e|liability|borrow|repay|payment due)']
```


```python
os_list = ['(third party|agreement|contract|arrangement|commitment|offshore|duty|tariff)']
```


```python
reg_list = ['(law|rule|compliance|government|congress|senate|certification|securities and exchange comission|requirement|federal|standard|generally accepted accounting principles)']
```

### Code


```python
sp500
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1957-03-04</td>
      <td>1800</td>
      <td>1888</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 8 columns</p>
</div>



- Calculate the firms’s buy and hold return over each time span, a la Assignment 2`b

- For each firm,
    - load the corresponding 10-K. Clean the text.
    - Create the sentiment measurements, and save those new measurements to the correct row and column in the dataframe.

        - Bonus: Save the total length of the document (# of words)
        - Bonus: Save the # of unique words (similar to total length)

- Calculate the two return measurements. Save those to the correct row and column in the dataframe

- Downloads 2021 accounting data (2021 ccm_cleaned.dta) from the data repo (possibly useful in analysis) and adds them to the dataset

- Save the whole thing to output/analysis_sample.csv


```python
## Overall Sentiment Scores
```


```python
#BHR pos

 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        BHRpos_score = len(re.findall(NEAR_regex(regex_BHR_pos),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'BHR Pos Sent'] = BHRpos_score
    
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
#BHR neg

 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        BHRneg_score = len(re.findall(NEAR_regex(regex_BHR_neg),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'BHR Neg Sent'] = BHRneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
#LM pos

 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        LMpos_score = len(re.findall(NEAR_regex(regex_LM_pos),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'LM Pos Sent'] = LMpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
#LM neg

 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        LMneg_score = len(re.findall(NEAR_regex(regex_LM_neg),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'LM Neg Sent'] = LMneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
sp500
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>BHR Pos Sent</th>
      <th>BHR Neg Sent</th>
      <th>LM Pos Sent</th>
      <th>LM Neg Sent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>0.000144</td>
      <td>0.000144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0.024460</td>
      <td>0.023602</td>
      <td>0.000118</td>
      <td>0.000118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1957-03-04</td>
      <td>1800</td>
      <td>1888</td>
      <td>0.021590</td>
      <td>0.024394</td>
      <td>0.000077</td>
      <td>0.000077</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0.019753</td>
      <td>0.022645</td>
      <td>0.000179</td>
      <td>0.000179</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0.027968</td>
      <td>0.023964</td>
      <td>0.000500</td>
      <td>0.000500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 12 columns</p>
</div>



### Debt Sentiment Scores


```python
debt_list
```




    ['(leverage|bonds|solvency|credit|d/e|liability|borrow|repay|payment due)']




```python
debt_BHR_pos = [debt_list[0], regex_BHR_pos[0]]
```


```python
debt_BHR_neg = [debt_list[0], regex_BHR_neg[0]]
```


```python
debt_LM_pos = [debt_list[0], regex_LM_pos[0]]
```


```python
debt_LM_neg =[debt_list[0], regex_LM_neg[0]]
```

Debt BHR Pos


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        debt_BHRpos_score = len(re.findall(NEAR_regex(debt_BHR_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Debt BHR Pos Sent'] = debt_BHRpos_score
    
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


debt BHR Neg


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        debt_BHRneg_score = len(re.findall(NEAR_regex(debt_BHR_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Debt BHR Neg Sent'] = debt_BHRneg_score
    
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


debt LM pos


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        debt_LMpos_score = len(re.findall(NEAR_regex(debt_LM_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Debt LM Pos Sent'] = debt_LMpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


debt LM neg


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        debt_LMneg_score = len(re.findall(NEAR_regex(debt_LM_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Debt LM neg Sent'] = debt_LMneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


### OS Sentiment Scores


```python
OS_BHR_pos = [os_list[0], regex_BHR_pos[0]]
```


```python
OS_BHR_neg = [os_list[0], regex_BHR_neg[0]]
```


```python
OS_LM_pos= [os_list[0], regex_LM_pos[0]]
```


```python
OS_LM_neg = [os_list[0], regex_LM_neg[0]]
```


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        OS_BHRpos_score = len(re.findall(NEAR_regex(OS_BHR_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'OS BHR Pos Sent'] = OS_BHRpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        OS_BHRneg_score = len(re.findall(NEAR_regex(OS_BHR_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'OS BHR Neg Sent'] = OS_BHRneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        OS_LMpos_score = len(re.findall(NEAR_regex(OS_LM_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'OS LM Pos Sent'] = OS_LMpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        OS_LMneg_score = len(re.findall(NEAR_regex(OS_LM_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'OS LM neg Sent'] = OS_LMneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


### Regulation Sentiment Scores


```python
reg_BHR_pos = [os_list[0], regex_BHR_pos[0]]
```


```python
reg_BHR_neg = [os_list[0], regex_BHR_neg[0]]
```


```python
reg_LM_pos = [os_list[0], regex_LM_pos[0]]
```


```python
reg_LM_neg = [os_list[0], regex_LM_neg[0]]
```


```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        reg_BHRpos_score = len(re.findall(NEAR_regex(reg_BHR_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'OS Reg Pos Sent'] = reg_BHRpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        reg_BHRneg_score = len(re.findall(NEAR_regex(reg_BHR_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Reg BHR Neg Sent'] = reg_BHRneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        reg_LMpos_score = len(re.findall(NEAR_regex(reg_LM_pos, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Reg LM Pos Sent'] = reg_LMpos_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN



```python
 # open the zip file 
with ZipFile('inputs/10k_files/10k_files.zip','r') as zipfolder:
    
    #get list of files in zipped folder
    file_list = zipfolder.namelist()

    for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']}") 

        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            
        fpath = possible_files[0] # the first match is the path to the file

    # open the file
        with zipfolder.open(fpath) as report_file:
            html = report_file.read().decode(encoding="utf-8")
#Clean html
        soup = BeautifulSoup(html, features='lxml-xml')
        for div in soup.find_all("div", {'style':'display:none'}): 
            div.decompose()

        lower = soup.get_text().lower() #make every letter lowercase
        no_punc = re.sub(r'\W',' ', lower) #any weird character replace with a space
        cleaned = re.sub(r'\s+',' ',no_punc).strip() #put one space between each word

# B.  measure the sentiment  

        reg_LMneg_score = len(re.findall(NEAR_regex(reg_LM_neg, 12,),cleaned))/ len(cleaned.split())
        sp500.loc[index, 'Reg LM neg Sent'] = reg_LMneg_score
```

    Index: 0, Symbol: MMM
    Index: 1, Symbol: AOS
    Index: 2, Symbol: ABT
    Index: 3, Symbol: ABBV
    Index: 4, Symbol: ACN


# Get 10K Dates


```python
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>BHR Pos Sent</th>
      <th>BHR Neg Sent</th>
      <th>...</th>
      <th>Debt LM Pos Sent</th>
      <th>Debt LM neg Sent</th>
      <th>OS BHR Pos Sent</th>
      <th>OS BHR Neg Sent</th>
      <th>OS LM Pos Sent</th>
      <th>OS LM neg Sent</th>
      <th>OS Reg Pos Sent</th>
      <th>Reg BHR Neg Sent</th>
      <th>Reg LM Pos Sent</th>
      <th>Reg LM neg Sent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.000065</td>
      <td>0.000065</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0.024460</td>
      <td>0.023602</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000266</td>
      <td>0.000148</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000266</td>
      <td>0.000148</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1957-03-04</td>
      <td>1800</td>
      <td>1888</td>
      <td>0.021590</td>
      <td>0.024394</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000826</td>
      <td>0.000519</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000826</td>
      <td>0.000519</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0.019753</td>
      <td>0.022645</td>
      <td>...</td>
      <td>0.000032</td>
      <td>0.000032</td>
      <td>0.000341</td>
      <td>0.000374</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000341</td>
      <td>0.000374</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0.027968</td>
      <td>0.023964</td>
      <td>...</td>
      <td>0.000019</td>
      <td>0.000019</td>
      <td>0.001155</td>
      <td>0.001251</td>
      <td>0.000019</td>
      <td>0.000019</td>
      <td>0.001155</td>
      <td>0.001251</td>
      <td>0.000019</td>
      <td>0.000019</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
from requests_html import HTMLSession
session = HTMLSession()
session.headers.update({'User-Agent':'Margaux Brennan mab923@lehigh.edu'})

for index, row in sp500[:5].iterrows(): 
    # print the row's index, and the ticker from the row
        print(f"Index: {index}, Symbol: {row['Symbol']},CIK: {row['CIK']}") 
        cik = {row['CIK']}
        
        # get a list of possible files for this firm
        firm_folder   = "sec-edgar-filings/" + row['Symbol'] + '/10-K/*/*.html'
        possible_files = fnmatch.filter(file_list, firm_folder) 
        if len(possible_files) == 0: 
            continue
            sleep(0.5)
            
        fpath = possible_files[0] # the first match is the path to the file

    # get accesion_number 
        anum = fpath.split('/')[3]
    
    #loop through url and find filing date
        url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{anum}-index.html'
        r = session.get(url)
        fdate = r.html.find('#formDiv > div.formContent > div:nth-child(1) > div:nth-child(2)', first=True).text
    
    #add filing date to df
        sp500.loc[index, 'Filing_Date'] = fdate
```

    Index: 0, Symbol: MMM,CIK: 66740
    Index: 1, Symbol: AOS,CIK: 91142
    Index: 2, Symbol: ABT,CIK: 1800
    Index: 3, Symbol: ABBV,CIK: 1551152
    Index: 4, Symbol: ACN,CIK: 1467373



```python
sp500
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>BHR Pos Sent</th>
      <th>BHR Neg Sent</th>
      <th>...</th>
      <th>Debt LM neg Sent</th>
      <th>OS BHR Pos Sent</th>
      <th>OS BHR Neg Sent</th>
      <th>OS LM Pos Sent</th>
      <th>OS LM neg Sent</th>
      <th>OS Reg Pos Sent</th>
      <th>Reg BHR Neg Sent</th>
      <th>Reg LM Pos Sent</th>
      <th>Reg LM neg Sent</th>
      <th>Filing_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.000065</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022-02-09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AOS</td>
      <td>A. O. Smith</td>
      <td>Industrials</td>
      <td>Building Products</td>
      <td>Milwaukee, Wisconsin</td>
      <td>2017-07-26</td>
      <td>91142</td>
      <td>1916</td>
      <td>0.024460</td>
      <td>0.023602</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000266</td>
      <td>0.000148</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000266</td>
      <td>0.000148</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022-02-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>North Chicago, Illinois</td>
      <td>1957-03-04</td>
      <td>1800</td>
      <td>1888</td>
      <td>0.021590</td>
      <td>0.024394</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000826</td>
      <td>0.000519</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000826</td>
      <td>0.000519</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBV</td>
      <td>AbbVie</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>North Chicago, Illinois</td>
      <td>2012-12-31</td>
      <td>1551152</td>
      <td>2013 (1888)</td>
      <td>0.019753</td>
      <td>0.022645</td>
      <td>...</td>
      <td>0.000032</td>
      <td>0.000341</td>
      <td>0.000374</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000341</td>
      <td>0.000374</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2022-02-18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture</td>
      <td>Information Technology</td>
      <td>IT Consulting &amp; Other Services</td>
      <td>Dublin, Ireland</td>
      <td>2011-07-06</td>
      <td>1467373</td>
      <td>1989</td>
      <td>0.027968</td>
      <td>0.023964</td>
      <td>...</td>
      <td>0.000019</td>
      <td>0.001155</td>
      <td>0.001251</td>
      <td>0.000019</td>
      <td>0.000019</td>
      <td>0.001155</td>
      <td>0.001251</td>
      <td>0.000019</td>
      <td>0.000019</td>
      <td>2022-10-12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>YUM</td>
      <td>Yum! Brands</td>
      <td>Consumer Discretionary</td>
      <td>Restaurants</td>
      <td>Louisville, Kentucky</td>
      <td>1997-10-06</td>
      <td>1041061</td>
      <td>1997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>499</th>
      <td>ZBRA</td>
      <td>Zebra Technologies</td>
      <td>Information Technology</td>
      <td>Electronic Equipment &amp; Instruments</td>
      <td>Lincolnshire, Illinois</td>
      <td>2019-12-23</td>
      <td>877212</td>
      <td>1969</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>500</th>
      <td>ZBH</td>
      <td>Zimmer Biomet</td>
      <td>Health Care</td>
      <td>Health Care Equipment</td>
      <td>Warsaw, Indiana</td>
      <td>2001-08-07</td>
      <td>1136869</td>
      <td>1927</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>501</th>
      <td>ZION</td>
      <td>Zions Bancorporation</td>
      <td>Financials</td>
      <td>Regional Banks</td>
      <td>Salt Lake City, Utah</td>
      <td>2001-06-22</td>
      <td>109380</td>
      <td>1873</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>502</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>503 rows × 25 columns</p>
</div>




```python
sret
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>date</th>
      <th>ret</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JJSF</td>
      <td>2021-12-01</td>
      <td>-0.011276</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JJSF</td>
      <td>2021-12-02</td>
      <td>0.030954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JJSF</td>
      <td>2021-12-03</td>
      <td>0.000287</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JJSF</td>
      <td>2021-12-06</td>
      <td>0.014362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JJSF</td>
      <td>2021-12-07</td>
      <td>0.012459</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2594044</th>
      <td>TSLA</td>
      <td>2022-12-23</td>
      <td>-0.017551</td>
    </tr>
    <tr>
      <th>2594045</th>
      <td>TSLA</td>
      <td>2022-12-27</td>
      <td>-0.114089</td>
    </tr>
    <tr>
      <th>2594046</th>
      <td>TSLA</td>
      <td>2022-12-28</td>
      <td>0.033089</td>
    </tr>
    <tr>
      <th>2594047</th>
      <td>TSLA</td>
      <td>2022-12-29</td>
      <td>0.080827</td>
    </tr>
    <tr>
      <th>2594048</th>
      <td>TSLA</td>
      <td>2022-12-30</td>
      <td>0.011164</td>
    </tr>
  </tbody>
</table>
<p>2594049 rows × 3 columns</p>
</div>




```python
sp500wrets = sp500.merge(sret.reset_index().rename(columns={'ticker':'Symbol'}), 
            on = 'Symbol',
            how = 'left',
            indicator = True,
            validate = '1:m')

```


```python
sp500wrets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Symbol</th>
      <th>Security</th>
      <th>GICS Sector</th>
      <th>GICS Sub-Industry</th>
      <th>Headquarters Location</th>
      <th>Date added</th>
      <th>CIK</th>
      <th>Founded</th>
      <th>BHR Pos Sent</th>
      <th>BHR Neg Sent</th>
      <th>...</th>
      <th>OS LM neg Sent</th>
      <th>OS Reg Pos Sent</th>
      <th>Reg BHR Neg Sent</th>
      <th>Reg LM Pos Sent</th>
      <th>Reg LM neg Sent</th>
      <th>Filing_Date</th>
      <th>index</th>
      <th>date</th>
      <th>ret</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022-02-09</td>
      <td>1576475.0</td>
      <td>2021-12-01</td>
      <td>0.004058</td>
      <td>both</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022-02-09</td>
      <td>1576476.0</td>
      <td>2021-12-02</td>
      <td>-0.002753</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022-02-09</td>
      <td>1576477.0</td>
      <td>2021-12-03</td>
      <td>0.013685</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022-02-09</td>
      <td>1576478.0</td>
      <td>2021-12-06</td>
      <td>0.026711</td>
      <td>both</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MMM</td>
      <td>3M</td>
      <td>Industrials</td>
      <td>Industrial Conglomerates</td>
      <td>Saint Paul, Minnesota</td>
      <td>1957-03-04</td>
      <td>66740</td>
      <td>1902</td>
      <td>0.025683</td>
      <td>0.031662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000262</td>
      <td>0.000432</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022-02-09</td>
      <td>1576479.0</td>
      <td>2021-12-07</td>
      <td>-0.003668</td>
      <td>both</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>136764</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181228.0</td>
      <td>2022-12-23</td>
      <td>0.005033</td>
      <td>both</td>
    </tr>
    <tr>
      <th>136765</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181229.0</td>
      <td>2022-12-27</td>
      <td>-0.003156</td>
      <td>both</td>
    </tr>
    <tr>
      <th>136766</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181230.0</td>
      <td>2022-12-28</td>
      <td>-0.010117</td>
      <td>both</td>
    </tr>
    <tr>
      <th>136767</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181231.0</td>
      <td>2022-12-29</td>
      <td>0.030035</td>
      <td>both</td>
    </tr>
    <tr>
      <th>136768</th>
      <td>ZTS</td>
      <td>Zoetis</td>
      <td>Health Care</td>
      <td>Pharmaceuticals</td>
      <td>Parsippany, New Jersey</td>
      <td>2013-06-21</td>
      <td>1555280</td>
      <td>1952</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>181232.0</td>
      <td>2022-12-30</td>
      <td>-0.010800</td>
      <td>both</td>
    </tr>
  </tbody>
</table>
<p>136769 rows × 29 columns</p>
</div>




```python
#merged['date_diff'] =  merged['date'] - merged['Filing_Date']

#import datetime

#for index, row in sp500wrets.iterrows(): 
    # print the row's index, and the ticker from the row
 #       print(f"Index: {index}, Symbol: {row['Symbol']},Filing_Date: {row['Filing_Date']}, date: {row['date']}, ret: {row['ret']}") 
        
  #      fdate = {row['Filing_Date']}
   #     rdate = {row['date']}

    # difference between dates in timedelta
    #    diff = rdate - fdate
     #   merged.loc[index, 'days_since_10k'] = diff.days     
        
#merged[:40]#
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
