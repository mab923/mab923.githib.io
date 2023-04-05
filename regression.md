---
layout: wide_default
---

```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col
```

## Part 1: Quick (and dirty) EDA

**TIP: Do this data exploration in a "scrap" file so you can explore quickly and messily.**

_We are going to use this dataset (`input_data2/housing_train.csv`) for the regression and ML assignments, as well as the prediction contest. The general focus will be on modelling the **Sale Price** (`v_SalePrice`)._

You should do the usual data exploration. 
- Sample basics: What is the unit of observation? What time spans are covered?
- Look for outliers, missing values, or data errors
- Note what variables are continuous or discrete numbers, which variables are categorical variables (and whether the categorical ordering is meaningful)     
- You should read up on what all the variables mean from the documentation in the data folder.
- Visually explore the relationship between `v_Sale_Price` and other variables.
  - For continuous variables - take note of whether the relationship seems linear or quadratic or polynomial
  - For categorical variables - maybe try a box plot for the various levels?
  - Take notes about what you find    

(Delete this cell that contains these instructions before submission, so that your submission starts with the "EDA" section below this.)      


```python
ht = pd.read_csv('input_data2/housing_train.csv')
```

## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important


```python
ht.describe().T.style.format('{:,.2f}')
```




<style type="text/css">
</style>
<table id="T_405b8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_405b8_level0_col0" class="col_heading level0 col0" >count</th>
      <th id="T_405b8_level0_col1" class="col_heading level0 col1" >mean</th>
      <th id="T_405b8_level0_col2" class="col_heading level0 col2" >std</th>
      <th id="T_405b8_level0_col3" class="col_heading level0 col3" >min</th>
      <th id="T_405b8_level0_col4" class="col_heading level0 col4" >25%</th>
      <th id="T_405b8_level0_col5" class="col_heading level0 col5" >50%</th>
      <th id="T_405b8_level0_col6" class="col_heading level0 col6" >75%</th>
      <th id="T_405b8_level0_col7" class="col_heading level0 col7" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_405b8_level0_row0" class="row_heading level0 row0" >v_MS_SubClass</th>
      <td id="T_405b8_row0_col0" class="data row0 col0" >1,941.00</td>
      <td id="T_405b8_row0_col1" class="data row0 col1" >58.09</td>
      <td id="T_405b8_row0_col2" class="data row0 col2" >42.95</td>
      <td id="T_405b8_row0_col3" class="data row0 col3" >20.00</td>
      <td id="T_405b8_row0_col4" class="data row0 col4" >20.00</td>
      <td id="T_405b8_row0_col5" class="data row0 col5" >50.00</td>
      <td id="T_405b8_row0_col6" class="data row0 col6" >70.00</td>
      <td id="T_405b8_row0_col7" class="data row0 col7" >190.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row1" class="row_heading level0 row1" >v_Lot_Frontage</th>
      <td id="T_405b8_row1_col0" class="data row1 col0" >1,620.00</td>
      <td id="T_405b8_row1_col1" class="data row1 col1" >69.30</td>
      <td id="T_405b8_row1_col2" class="data row1 col2" >23.98</td>
      <td id="T_405b8_row1_col3" class="data row1 col3" >21.00</td>
      <td id="T_405b8_row1_col4" class="data row1 col4" >58.00</td>
      <td id="T_405b8_row1_col5" class="data row1 col5" >68.00</td>
      <td id="T_405b8_row1_col6" class="data row1 col6" >80.00</td>
      <td id="T_405b8_row1_col7" class="data row1 col7" >313.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row2" class="row_heading level0 row2" >v_Lot_Area</th>
      <td id="T_405b8_row2_col0" class="data row2 col0" >1,941.00</td>
      <td id="T_405b8_row2_col1" class="data row2 col1" >10,284.77</td>
      <td id="T_405b8_row2_col2" class="data row2 col2" >7,832.30</td>
      <td id="T_405b8_row2_col3" class="data row2 col3" >1,470.00</td>
      <td id="T_405b8_row2_col4" class="data row2 col4" >7,420.00</td>
      <td id="T_405b8_row2_col5" class="data row2 col5" >9,450.00</td>
      <td id="T_405b8_row2_col6" class="data row2 col6" >11,631.00</td>
      <td id="T_405b8_row2_col7" class="data row2 col7" >164,660.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row3" class="row_heading level0 row3" >v_Overall_Qual</th>
      <td id="T_405b8_row3_col0" class="data row3 col0" >1,941.00</td>
      <td id="T_405b8_row3_col1" class="data row3 col1" >6.11</td>
      <td id="T_405b8_row3_col2" class="data row3 col2" >1.40</td>
      <td id="T_405b8_row3_col3" class="data row3 col3" >1.00</td>
      <td id="T_405b8_row3_col4" class="data row3 col4" >5.00</td>
      <td id="T_405b8_row3_col5" class="data row3 col5" >6.00</td>
      <td id="T_405b8_row3_col6" class="data row3 col6" >7.00</td>
      <td id="T_405b8_row3_col7" class="data row3 col7" >10.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row4" class="row_heading level0 row4" >v_Overall_Cond</th>
      <td id="T_405b8_row4_col0" class="data row4 col0" >1,941.00</td>
      <td id="T_405b8_row4_col1" class="data row4 col1" >5.57</td>
      <td id="T_405b8_row4_col2" class="data row4 col2" >1.09</td>
      <td id="T_405b8_row4_col3" class="data row4 col3" >1.00</td>
      <td id="T_405b8_row4_col4" class="data row4 col4" >5.00</td>
      <td id="T_405b8_row4_col5" class="data row4 col5" >5.00</td>
      <td id="T_405b8_row4_col6" class="data row4 col6" >6.00</td>
      <td id="T_405b8_row4_col7" class="data row4 col7" >9.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row5" class="row_heading level0 row5" >v_Year_Built</th>
      <td id="T_405b8_row5_col0" class="data row5 col0" >1,941.00</td>
      <td id="T_405b8_row5_col1" class="data row5 col1" >1,971.32</td>
      <td id="T_405b8_row5_col2" class="data row5 col2" >30.21</td>
      <td id="T_405b8_row5_col3" class="data row5 col3" >1,872.00</td>
      <td id="T_405b8_row5_col4" class="data row5 col4" >1,953.00</td>
      <td id="T_405b8_row5_col5" class="data row5 col5" >1,973.00</td>
      <td id="T_405b8_row5_col6" class="data row5 col6" >2,001.00</td>
      <td id="T_405b8_row5_col7" class="data row5 col7" >2,008.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row6" class="row_heading level0 row6" >v_Year_Remod/Add</th>
      <td id="T_405b8_row6_col0" class="data row6 col0" >1,941.00</td>
      <td id="T_405b8_row6_col1" class="data row6 col1" >1,984.07</td>
      <td id="T_405b8_row6_col2" class="data row6 col2" >20.84</td>
      <td id="T_405b8_row6_col3" class="data row6 col3" >1,950.00</td>
      <td id="T_405b8_row6_col4" class="data row6 col4" >1,965.00</td>
      <td id="T_405b8_row6_col5" class="data row6 col5" >1,993.00</td>
      <td id="T_405b8_row6_col6" class="data row6 col6" >2,004.00</td>
      <td id="T_405b8_row6_col7" class="data row6 col7" >2,009.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row7" class="row_heading level0 row7" >v_Mas_Vnr_Area</th>
      <td id="T_405b8_row7_col0" class="data row7 col0" >1,923.00</td>
      <td id="T_405b8_row7_col1" class="data row7 col1" >104.85</td>
      <td id="T_405b8_row7_col2" class="data row7 col2" >184.98</td>
      <td id="T_405b8_row7_col3" class="data row7 col3" >0.00</td>
      <td id="T_405b8_row7_col4" class="data row7 col4" >0.00</td>
      <td id="T_405b8_row7_col5" class="data row7 col5" >0.00</td>
      <td id="T_405b8_row7_col6" class="data row7 col6" >168.00</td>
      <td id="T_405b8_row7_col7" class="data row7 col7" >1,600.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row8" class="row_heading level0 row8" >v_BsmtFin_SF_1</th>
      <td id="T_405b8_row8_col0" class="data row8 col0" >1,940.00</td>
      <td id="T_405b8_row8_col1" class="data row8 col1" >436.99</td>
      <td id="T_405b8_row8_col2" class="data row8 col2" >457.82</td>
      <td id="T_405b8_row8_col3" class="data row8 col3" >0.00</td>
      <td id="T_405b8_row8_col4" class="data row8 col4" >0.00</td>
      <td id="T_405b8_row8_col5" class="data row8 col5" >361.50</td>
      <td id="T_405b8_row8_col6" class="data row8 col6" >735.25</td>
      <td id="T_405b8_row8_col7" class="data row8 col7" >5,644.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row9" class="row_heading level0 row9" >v_BsmtFin_SF_2</th>
      <td id="T_405b8_row9_col0" class="data row9 col0" >1,940.00</td>
      <td id="T_405b8_row9_col1" class="data row9 col1" >49.25</td>
      <td id="T_405b8_row9_col2" class="data row9 col2" >169.56</td>
      <td id="T_405b8_row9_col3" class="data row9 col3" >0.00</td>
      <td id="T_405b8_row9_col4" class="data row9 col4" >0.00</td>
      <td id="T_405b8_row9_col5" class="data row9 col5" >0.00</td>
      <td id="T_405b8_row9_col6" class="data row9 col6" >0.00</td>
      <td id="T_405b8_row9_col7" class="data row9 col7" >1,474.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row10" class="row_heading level0 row10" >v_Bsmt_Unf_SF</th>
      <td id="T_405b8_row10_col0" class="data row10 col0" >1,940.00</td>
      <td id="T_405b8_row10_col1" class="data row10 col1" >567.44</td>
      <td id="T_405b8_row10_col2" class="data row10 col2" >439.60</td>
      <td id="T_405b8_row10_col3" class="data row10 col3" >0.00</td>
      <td id="T_405b8_row10_col4" class="data row10 col4" >225.75</td>
      <td id="T_405b8_row10_col5" class="data row10 col5" >474.00</td>
      <td id="T_405b8_row10_col6" class="data row10 col6" >815.00</td>
      <td id="T_405b8_row10_col7" class="data row10 col7" >2,153.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row11" class="row_heading level0 row11" >v_Total_Bsmt_SF</th>
      <td id="T_405b8_row11_col0" class="data row11 col0" >1,940.00</td>
      <td id="T_405b8_row11_col1" class="data row11 col1" >1,053.67</td>
      <td id="T_405b8_row11_col2" class="data row11 col2" >438.66</td>
      <td id="T_405b8_row11_col3" class="data row11 col3" >0.00</td>
      <td id="T_405b8_row11_col4" class="data row11 col4" >796.75</td>
      <td id="T_405b8_row11_col5" class="data row11 col5" >989.50</td>
      <td id="T_405b8_row11_col6" class="data row11 col6" >1,295.25</td>
      <td id="T_405b8_row11_col7" class="data row11 col7" >6,110.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row12" class="row_heading level0 row12" >v_1st_Flr_SF</th>
      <td id="T_405b8_row12_col0" class="data row12 col0" >1,941.00</td>
      <td id="T_405b8_row12_col1" class="data row12 col1" >1,161.07</td>
      <td id="T_405b8_row12_col2" class="data row12 col2" >396.95</td>
      <td id="T_405b8_row12_col3" class="data row12 col3" >334.00</td>
      <td id="T_405b8_row12_col4" class="data row12 col4" >886.00</td>
      <td id="T_405b8_row12_col5" class="data row12 col5" >1,085.00</td>
      <td id="T_405b8_row12_col6" class="data row12 col6" >1,383.00</td>
      <td id="T_405b8_row12_col7" class="data row12 col7" >5,095.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row13" class="row_heading level0 row13" >v_2nd_Flr_SF</th>
      <td id="T_405b8_row13_col0" class="data row13 col0" >1,941.00</td>
      <td id="T_405b8_row13_col1" class="data row13 col1" >340.96</td>
      <td id="T_405b8_row13_col2" class="data row13 col2" >434.24</td>
      <td id="T_405b8_row13_col3" class="data row13 col3" >0.00</td>
      <td id="T_405b8_row13_col4" class="data row13 col4" >0.00</td>
      <td id="T_405b8_row13_col5" class="data row13 col5" >0.00</td>
      <td id="T_405b8_row13_col6" class="data row13 col6" >717.00</td>
      <td id="T_405b8_row13_col7" class="data row13 col7" >2,065.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row14" class="row_heading level0 row14" >v_Low_Qual_Fin_SF</th>
      <td id="T_405b8_row14_col0" class="data row14 col0" >1,941.00</td>
      <td id="T_405b8_row14_col1" class="data row14 col1" >4.28</td>
      <td id="T_405b8_row14_col2" class="data row14 col2" >42.94</td>
      <td id="T_405b8_row14_col3" class="data row14 col3" >0.00</td>
      <td id="T_405b8_row14_col4" class="data row14 col4" >0.00</td>
      <td id="T_405b8_row14_col5" class="data row14 col5" >0.00</td>
      <td id="T_405b8_row14_col6" class="data row14 col6" >0.00</td>
      <td id="T_405b8_row14_col7" class="data row14 col7" >697.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row15" class="row_heading level0 row15" >v_Gr_Liv_Area</th>
      <td id="T_405b8_row15_col0" class="data row15 col0" >1,941.00</td>
      <td id="T_405b8_row15_col1" class="data row15 col1" >1,506.31</td>
      <td id="T_405b8_row15_col2" class="data row15 col2" >524.77</td>
      <td id="T_405b8_row15_col3" class="data row15 col3" >334.00</td>
      <td id="T_405b8_row15_col4" class="data row15 col4" >1,118.00</td>
      <td id="T_405b8_row15_col5" class="data row15 col5" >1,436.00</td>
      <td id="T_405b8_row15_col6" class="data row15 col6" >1,755.00</td>
      <td id="T_405b8_row15_col7" class="data row15 col7" >5,642.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row16" class="row_heading level0 row16" >v_Bsmt_Full_Bath</th>
      <td id="T_405b8_row16_col0" class="data row16 col0" >1,939.00</td>
      <td id="T_405b8_row16_col1" class="data row16 col1" >0.42</td>
      <td id="T_405b8_row16_col2" class="data row16 col2" >0.52</td>
      <td id="T_405b8_row16_col3" class="data row16 col3" >0.00</td>
      <td id="T_405b8_row16_col4" class="data row16 col4" >0.00</td>
      <td id="T_405b8_row16_col5" class="data row16 col5" >0.00</td>
      <td id="T_405b8_row16_col6" class="data row16 col6" >1.00</td>
      <td id="T_405b8_row16_col7" class="data row16 col7" >2.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row17" class="row_heading level0 row17" >v_Bsmt_Half_Bath</th>
      <td id="T_405b8_row17_col0" class="data row17 col0" >1,939.00</td>
      <td id="T_405b8_row17_col1" class="data row17 col1" >0.06</td>
      <td id="T_405b8_row17_col2" class="data row17 col2" >0.25</td>
      <td id="T_405b8_row17_col3" class="data row17 col3" >0.00</td>
      <td id="T_405b8_row17_col4" class="data row17 col4" >0.00</td>
      <td id="T_405b8_row17_col5" class="data row17 col5" >0.00</td>
      <td id="T_405b8_row17_col6" class="data row17 col6" >0.00</td>
      <td id="T_405b8_row17_col7" class="data row17 col7" >2.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row18" class="row_heading level0 row18" >v_Full_Bath</th>
      <td id="T_405b8_row18_col0" class="data row18 col0" >1,941.00</td>
      <td id="T_405b8_row18_col1" class="data row18 col1" >1.57</td>
      <td id="T_405b8_row18_col2" class="data row18 col2" >0.55</td>
      <td id="T_405b8_row18_col3" class="data row18 col3" >0.00</td>
      <td id="T_405b8_row18_col4" class="data row18 col4" >1.00</td>
      <td id="T_405b8_row18_col5" class="data row18 col5" >2.00</td>
      <td id="T_405b8_row18_col6" class="data row18 col6" >2.00</td>
      <td id="T_405b8_row18_col7" class="data row18 col7" >3.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row19" class="row_heading level0 row19" >v_Half_Bath</th>
      <td id="T_405b8_row19_col0" class="data row19 col0" >1,941.00</td>
      <td id="T_405b8_row19_col1" class="data row19 col1" >0.38</td>
      <td id="T_405b8_row19_col2" class="data row19 col2" >0.50</td>
      <td id="T_405b8_row19_col3" class="data row19 col3" >0.00</td>
      <td id="T_405b8_row19_col4" class="data row19 col4" >0.00</td>
      <td id="T_405b8_row19_col5" class="data row19 col5" >0.00</td>
      <td id="T_405b8_row19_col6" class="data row19 col6" >1.00</td>
      <td id="T_405b8_row19_col7" class="data row19 col7" >2.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row20" class="row_heading level0 row20" >v_Bedroom_AbvGr</th>
      <td id="T_405b8_row20_col0" class="data row20 col0" >1,941.00</td>
      <td id="T_405b8_row20_col1" class="data row20 col1" >2.87</td>
      <td id="T_405b8_row20_col2" class="data row20 col2" >0.83</td>
      <td id="T_405b8_row20_col3" class="data row20 col3" >0.00</td>
      <td id="T_405b8_row20_col4" class="data row20 col4" >2.00</td>
      <td id="T_405b8_row20_col5" class="data row20 col5" >3.00</td>
      <td id="T_405b8_row20_col6" class="data row20 col6" >3.00</td>
      <td id="T_405b8_row20_col7" class="data row20 col7" >8.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row21" class="row_heading level0 row21" >v_Kitchen_AbvGr</th>
      <td id="T_405b8_row21_col0" class="data row21 col0" >1,941.00</td>
      <td id="T_405b8_row21_col1" class="data row21 col1" >1.04</td>
      <td id="T_405b8_row21_col2" class="data row21 col2" >0.20</td>
      <td id="T_405b8_row21_col3" class="data row21 col3" >0.00</td>
      <td id="T_405b8_row21_col4" class="data row21 col4" >1.00</td>
      <td id="T_405b8_row21_col5" class="data row21 col5" >1.00</td>
      <td id="T_405b8_row21_col6" class="data row21 col6" >1.00</td>
      <td id="T_405b8_row21_col7" class="data row21 col7" >2.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row22" class="row_heading level0 row22" >v_TotRms_AbvGrd</th>
      <td id="T_405b8_row22_col0" class="data row22 col0" >1,941.00</td>
      <td id="T_405b8_row22_col1" class="data row22 col1" >6.47</td>
      <td id="T_405b8_row22_col2" class="data row22 col2" >1.58</td>
      <td id="T_405b8_row22_col3" class="data row22 col3" >2.00</td>
      <td id="T_405b8_row22_col4" class="data row22 col4" >5.00</td>
      <td id="T_405b8_row22_col5" class="data row22 col5" >6.00</td>
      <td id="T_405b8_row22_col6" class="data row22 col6" >7.00</td>
      <td id="T_405b8_row22_col7" class="data row22 col7" >15.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row23" class="row_heading level0 row23" >v_Fireplaces</th>
      <td id="T_405b8_row23_col0" class="data row23 col0" >1,941.00</td>
      <td id="T_405b8_row23_col1" class="data row23 col1" >0.60</td>
      <td id="T_405b8_row23_col2" class="data row23 col2" >0.64</td>
      <td id="T_405b8_row23_col3" class="data row23 col3" >0.00</td>
      <td id="T_405b8_row23_col4" class="data row23 col4" >0.00</td>
      <td id="T_405b8_row23_col5" class="data row23 col5" >1.00</td>
      <td id="T_405b8_row23_col6" class="data row23 col6" >1.00</td>
      <td id="T_405b8_row23_col7" class="data row23 col7" >4.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row24" class="row_heading level0 row24" >v_Garage_Yr_Blt</th>
      <td id="T_405b8_row24_col0" class="data row24 col0" >1,834.00</td>
      <td id="T_405b8_row24_col1" class="data row24 col1" >1,978.19</td>
      <td id="T_405b8_row24_col2" class="data row24 col2" >25.73</td>
      <td id="T_405b8_row24_col3" class="data row24 col3" >1,895.00</td>
      <td id="T_405b8_row24_col4" class="data row24 col4" >1,960.00</td>
      <td id="T_405b8_row24_col5" class="data row24 col5" >1,980.00</td>
      <td id="T_405b8_row24_col6" class="data row24 col6" >2,002.00</td>
      <td id="T_405b8_row24_col7" class="data row24 col7" >2,207.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row25" class="row_heading level0 row25" >v_Garage_Cars</th>
      <td id="T_405b8_row25_col0" class="data row25 col0" >1,940.00</td>
      <td id="T_405b8_row25_col1" class="data row25 col1" >1.77</td>
      <td id="T_405b8_row25_col2" class="data row25 col2" >0.76</td>
      <td id="T_405b8_row25_col3" class="data row25 col3" >0.00</td>
      <td id="T_405b8_row25_col4" class="data row25 col4" >1.00</td>
      <td id="T_405b8_row25_col5" class="data row25 col5" >2.00</td>
      <td id="T_405b8_row25_col6" class="data row25 col6" >2.00</td>
      <td id="T_405b8_row25_col7" class="data row25 col7" >4.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row26" class="row_heading level0 row26" >v_Garage_Area</th>
      <td id="T_405b8_row26_col0" class="data row26 col0" >1,940.00</td>
      <td id="T_405b8_row26_col1" class="data row26 col1" >472.77</td>
      <td id="T_405b8_row26_col2" class="data row26 col2" >217.09</td>
      <td id="T_405b8_row26_col3" class="data row26 col3" >0.00</td>
      <td id="T_405b8_row26_col4" class="data row26 col4" >318.75</td>
      <td id="T_405b8_row26_col5" class="data row26 col5" >478.00</td>
      <td id="T_405b8_row26_col6" class="data row26 col6" >576.00</td>
      <td id="T_405b8_row26_col7" class="data row26 col7" >1,488.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row27" class="row_heading level0 row27" >v_Wood_Deck_SF</th>
      <td id="T_405b8_row27_col0" class="data row27 col0" >1,941.00</td>
      <td id="T_405b8_row27_col1" class="data row27 col1" >92.46</td>
      <td id="T_405b8_row27_col2" class="data row27 col2" >127.02</td>
      <td id="T_405b8_row27_col3" class="data row27 col3" >0.00</td>
      <td id="T_405b8_row27_col4" class="data row27 col4" >0.00</td>
      <td id="T_405b8_row27_col5" class="data row27 col5" >0.00</td>
      <td id="T_405b8_row27_col6" class="data row27 col6" >168.00</td>
      <td id="T_405b8_row27_col7" class="data row27 col7" >1,424.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row28" class="row_heading level0 row28" >v_Open_Porch_SF</th>
      <td id="T_405b8_row28_col0" class="data row28 col0" >1,941.00</td>
      <td id="T_405b8_row28_col1" class="data row28 col1" >49.16</td>
      <td id="T_405b8_row28_col2" class="data row28 col2" >70.30</td>
      <td id="T_405b8_row28_col3" class="data row28 col3" >0.00</td>
      <td id="T_405b8_row28_col4" class="data row28 col4" >0.00</td>
      <td id="T_405b8_row28_col5" class="data row28 col5" >28.00</td>
      <td id="T_405b8_row28_col6" class="data row28 col6" >72.00</td>
      <td id="T_405b8_row28_col7" class="data row28 col7" >742.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row29" class="row_heading level0 row29" >v_Enclosed_Porch</th>
      <td id="T_405b8_row29_col0" class="data row29 col0" >1,941.00</td>
      <td id="T_405b8_row29_col1" class="data row29 col1" >22.95</td>
      <td id="T_405b8_row29_col2" class="data row29 col2" >65.25</td>
      <td id="T_405b8_row29_col3" class="data row29 col3" >0.00</td>
      <td id="T_405b8_row29_col4" class="data row29 col4" >0.00</td>
      <td id="T_405b8_row29_col5" class="data row29 col5" >0.00</td>
      <td id="T_405b8_row29_col6" class="data row29 col6" >0.00</td>
      <td id="T_405b8_row29_col7" class="data row29 col7" >1,012.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row30" class="row_heading level0 row30" >v_3Ssn_Porch</th>
      <td id="T_405b8_row30_col0" class="data row30 col0" >1,941.00</td>
      <td id="T_405b8_row30_col1" class="data row30 col1" >2.25</td>
      <td id="T_405b8_row30_col2" class="data row30 col2" >22.42</td>
      <td id="T_405b8_row30_col3" class="data row30 col3" >0.00</td>
      <td id="T_405b8_row30_col4" class="data row30 col4" >0.00</td>
      <td id="T_405b8_row30_col5" class="data row30 col5" >0.00</td>
      <td id="T_405b8_row30_col6" class="data row30 col6" >0.00</td>
      <td id="T_405b8_row30_col7" class="data row30 col7" >407.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row31" class="row_heading level0 row31" >v_Screen_Porch</th>
      <td id="T_405b8_row31_col0" class="data row31 col0" >1,941.00</td>
      <td id="T_405b8_row31_col1" class="data row31 col1" >16.25</td>
      <td id="T_405b8_row31_col2" class="data row31 col2" >56.75</td>
      <td id="T_405b8_row31_col3" class="data row31 col3" >0.00</td>
      <td id="T_405b8_row31_col4" class="data row31 col4" >0.00</td>
      <td id="T_405b8_row31_col5" class="data row31 col5" >0.00</td>
      <td id="T_405b8_row31_col6" class="data row31 col6" >0.00</td>
      <td id="T_405b8_row31_col7" class="data row31 col7" >576.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row32" class="row_heading level0 row32" >v_Pool_Area</th>
      <td id="T_405b8_row32_col0" class="data row32 col0" >1,941.00</td>
      <td id="T_405b8_row32_col1" class="data row32 col1" >3.39</td>
      <td id="T_405b8_row32_col2" class="data row32 col2" >43.70</td>
      <td id="T_405b8_row32_col3" class="data row32 col3" >0.00</td>
      <td id="T_405b8_row32_col4" class="data row32 col4" >0.00</td>
      <td id="T_405b8_row32_col5" class="data row32 col5" >0.00</td>
      <td id="T_405b8_row32_col6" class="data row32 col6" >0.00</td>
      <td id="T_405b8_row32_col7" class="data row32 col7" >800.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row33" class="row_heading level0 row33" >v_Misc_Val</th>
      <td id="T_405b8_row33_col0" class="data row33 col0" >1,941.00</td>
      <td id="T_405b8_row33_col1" class="data row33 col1" >52.55</td>
      <td id="T_405b8_row33_col2" class="data row33 col2" >616.06</td>
      <td id="T_405b8_row33_col3" class="data row33 col3" >0.00</td>
      <td id="T_405b8_row33_col4" class="data row33 col4" >0.00</td>
      <td id="T_405b8_row33_col5" class="data row33 col5" >0.00</td>
      <td id="T_405b8_row33_col6" class="data row33 col6" >0.00</td>
      <td id="T_405b8_row33_col7" class="data row33 col7" >17,000.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row34" class="row_heading level0 row34" >v_Mo_Sold</th>
      <td id="T_405b8_row34_col0" class="data row34 col0" >1,941.00</td>
      <td id="T_405b8_row34_col1" class="data row34 col1" >6.43</td>
      <td id="T_405b8_row34_col2" class="data row34 col2" >2.75</td>
      <td id="T_405b8_row34_col3" class="data row34 col3" >1.00</td>
      <td id="T_405b8_row34_col4" class="data row34 col4" >5.00</td>
      <td id="T_405b8_row34_col5" class="data row34 col5" >6.00</td>
      <td id="T_405b8_row34_col6" class="data row34 col6" >8.00</td>
      <td id="T_405b8_row34_col7" class="data row34 col7" >12.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row35" class="row_heading level0 row35" >v_Yr_Sold</th>
      <td id="T_405b8_row35_col0" class="data row35 col0" >1,941.00</td>
      <td id="T_405b8_row35_col1" class="data row35 col1" >2,007.00</td>
      <td id="T_405b8_row35_col2" class="data row35 col2" >0.80</td>
      <td id="T_405b8_row35_col3" class="data row35 col3" >2,006.00</td>
      <td id="T_405b8_row35_col4" class="data row35 col4" >2,006.00</td>
      <td id="T_405b8_row35_col5" class="data row35 col5" >2,007.00</td>
      <td id="T_405b8_row35_col6" class="data row35 col6" >2,008.00</td>
      <td id="T_405b8_row35_col7" class="data row35 col7" >2,008.00</td>
    </tr>
    <tr>
      <th id="T_405b8_level0_row36" class="row_heading level0 row36" >v_SalePrice</th>
      <td id="T_405b8_row36_col0" class="data row36 col0" >1,941.00</td>
      <td id="T_405b8_row36_col1" class="data row36 col1" >182,033.24</td>
      <td id="T_405b8_row36_col2" class="data row36 col2" >80,407.10</td>
      <td id="T_405b8_row36_col3" class="data row36 col3" >13,100.00</td>
      <td id="T_405b8_row36_col4" class="data row36 col4" >130,000.00</td>
      <td id="T_405b8_row36_col5" class="data row36 col5" >161,900.00</td>
      <td id="T_405b8_row36_col6" class="data row36 col6" >215,000.00</td>
      <td id="T_405b8_row36_col7" class="data row36 col7" >755,000.00</td>
    </tr>
  </tbody>
</table>





```python
print(ht.head(),  '\n---')
print(ht.tail(),  '\n---')
print(ht.columns, '\n---')
print("The shape is: ",ht.shape, '\n---')
print("Info:",ht.info(), '\n---') # memory usage, name, dtype, and # of non-null obs (--> # of missing obs) per variable
#print(iris.describe(), '\n---') # summary stats, and you can customize the list!
print(ht['v_SalePrice'].value_counts()[:10], '\n---')
print(ht['v_SalePrice'].nunique(), '\n---')
```

               parcel  v_MS_SubClass v_MS_Zoning  v_Lot_Frontage  v_Lot_Area  \
    0  1056_528110080             20          RL           107.0       13891   
    1  1055_528108150             20          RL            98.0       12704   
    2  1053_528104050             20          RL           114.0       14803   
    3  2213_909275160             20          RL           126.0       13108   
    4  1051_528102030             20          RL            96.0       12444   
    
      v_Street v_Alley v_Lot_Shape v_Land_Contour v_Utilities  ... v_Pool_Area  \
    0     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    2     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    3     Pave     NaN         IR2            HLS      AllPub  ...           0   
    4     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    
      v_Pool_QC v_Fence v_Misc_Feature v_Misc_Val v_Mo_Sold v_Yr_Sold  \
    0       NaN     NaN            NaN          0         1      2008   
    1       NaN     NaN            NaN          0         1      2008   
    2       NaN     NaN            NaN          0         6      2008   
    3       NaN     NaN            NaN          0         6      2007   
    4       NaN     NaN            NaN          0        11      2008   
    
       v_Sale_Type  v_Sale_Condition  v_SalePrice  
    0          New           Partial       372402  
    1          New           Partial       317500  
    2          New           Partial       385000  
    3          WD             Normal       153500  
    4          New           Partial       394617  
    
    [5 rows x 81 columns] 
    ---
                  parcel  v_MS_SubClass v_MS_Zoning  v_Lot_Frontage  v_Lot_Area  \
    1936  2524_534125210            190          RL            79.0       13110   
    1937  2846_909131125            190          RH             NaN        7082   
    1938  2605_535382020            190          RL            60.0       10800   
    1939  1516_909101180            190          RL            55.0        5687   
    1940  1387_905200100            190          RL            60.0       12900   
    
         v_Street v_Alley v_Lot_Shape v_Land_Contour v_Utilities  ... v_Pool_Area  \
    1936     Pave     NaN         IR1            Lvl      AllPub  ...           0   
    1937     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1938     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1939     Pave    Grvl         Reg            Bnk      AllPub  ...           0   
    1940     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    
         v_Pool_QC v_Fence v_Misc_Feature v_Misc_Val v_Mo_Sold v_Yr_Sold  \
    1936       NaN   MnPrv            NaN          0         7      2006   
    1937       NaN     NaN            NaN          0         7      2006   
    1938       NaN     NaN            NaN          0         5      2006   
    1939       NaN     NaN            NaN          0         3      2008   
    1940       NaN     NaN            NaN          0         1      2008   
    
          v_Sale_Type  v_Sale_Condition  v_SalePrice  
    1936          WD             Normal       146500  
    1937          WD             Normal       160000  
    1938        ConLD            Normal       160000  
    1939          WD             Normal       135900  
    1940          WD             Alloca        95541  
    
    [5 rows x 81 columns] 
    ---
    Index(['parcel', 'v_MS_SubClass', 'v_MS_Zoning', 'v_Lot_Frontage',
           'v_Lot_Area', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour',
           'v_Utilities', 'v_Lot_Config', 'v_Land_Slope', 'v_Neighborhood',
           'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
           'v_Overall_Qual', 'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add',
           'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd',
           'v_Mas_Vnr_Type', 'v_Mas_Vnr_Area', 'v_Exter_Qual', 'v_Exter_Cond',
           'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
           'v_BsmtFin_Type_1', 'v_BsmtFin_SF_1', 'v_BsmtFin_Type_2',
           'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_Heating',
           'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_1st_Flr_SF',
           'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
           'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
           'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_Kitchen_Qual',
           'v_TotRms_AbvGrd', 'v_Functional', 'v_Fireplaces', 'v_Fireplace_Qu',
           'v_Garage_Type', 'v_Garage_Yr_Blt', 'v_Garage_Finish', 'v_Garage_Cars',
           'v_Garage_Area', 'v_Garage_Qual', 'v_Garage_Cond', 'v_Paved_Drive',
           'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch',
           'v_Screen_Porch', 'v_Pool_Area', 'v_Pool_QC', 'v_Fence',
           'v_Misc_Feature', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_Sale_Type',
           'v_Sale_Condition', 'v_SalePrice'],
          dtype='object') 
    ---
    The shape is:  (1941, 81) 
    ---
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1941 entries, 0 to 1940
    Data columns (total 81 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   parcel             1941 non-null   object 
     1   v_MS_SubClass      1941 non-null   int64  
     2   v_MS_Zoning        1941 non-null   object 
     3   v_Lot_Frontage     1620 non-null   float64
     4   v_Lot_Area         1941 non-null   int64  
     5   v_Street           1941 non-null   object 
     6   v_Alley            136 non-null    object 
     7   v_Lot_Shape        1941 non-null   object 
     8   v_Land_Contour     1941 non-null   object 
     9   v_Utilities        1941 non-null   object 
     10  v_Lot_Config       1941 non-null   object 
     11  v_Land_Slope       1941 non-null   object 
     12  v_Neighborhood     1941 non-null   object 
     13  v_Condition_1      1941 non-null   object 
     14  v_Condition_2      1941 non-null   object 
     15  v_Bldg_Type        1941 non-null   object 
     16  v_House_Style      1941 non-null   object 
     17  v_Overall_Qual     1941 non-null   int64  
     18  v_Overall_Cond     1941 non-null   int64  
     19  v_Year_Built       1941 non-null   int64  
     20  v_Year_Remod/Add   1941 non-null   int64  
     21  v_Roof_Style       1941 non-null   object 
     22  v_Roof_Matl        1941 non-null   object 
     23  v_Exterior_1st     1941 non-null   object 
     24  v_Exterior_2nd     1941 non-null   object 
     25  v_Mas_Vnr_Type     1923 non-null   object 
     26  v_Mas_Vnr_Area     1923 non-null   float64
     27  v_Exter_Qual       1941 non-null   object 
     28  v_Exter_Cond       1941 non-null   object 
     29  v_Foundation       1941 non-null   object 
     30  v_Bsmt_Qual        1891 non-null   object 
     31  v_Bsmt_Cond        1891 non-null   object 
     32  v_Bsmt_Exposure    1889 non-null   object 
     33  v_BsmtFin_Type_1   1891 non-null   object 
     34  v_BsmtFin_SF_1     1940 non-null   float64
     35  v_BsmtFin_Type_2   1891 non-null   object 
     36  v_BsmtFin_SF_2     1940 non-null   float64
     37  v_Bsmt_Unf_SF      1940 non-null   float64
     38  v_Total_Bsmt_SF    1940 non-null   float64
     39  v_Heating          1941 non-null   object 
     40  v_Heating_QC       1941 non-null   object 
     41  v_Central_Air      1941 non-null   object 
     42  v_Electrical       1940 non-null   object 
     43  v_1st_Flr_SF       1941 non-null   int64  
     44  v_2nd_Flr_SF       1941 non-null   int64  
     45  v_Low_Qual_Fin_SF  1941 non-null   int64  
     46  v_Gr_Liv_Area      1941 non-null   int64  
     47  v_Bsmt_Full_Bath   1939 non-null   float64
     48  v_Bsmt_Half_Bath   1939 non-null   float64
     49  v_Full_Bath        1941 non-null   int64  
     50  v_Half_Bath        1941 non-null   int64  
     51  v_Bedroom_AbvGr    1941 non-null   int64  
     52  v_Kitchen_AbvGr    1941 non-null   int64  
     53  v_Kitchen_Qual     1941 non-null   object 
     54  v_TotRms_AbvGrd    1941 non-null   int64  
     55  v_Functional       1941 non-null   object 
     56  v_Fireplaces       1941 non-null   int64  
     57  v_Fireplace_Qu     1001 non-null   object 
     58  v_Garage_Type      1836 non-null   object 
     59  v_Garage_Yr_Blt    1834 non-null   float64
     60  v_Garage_Finish    1834 non-null   object 
     61  v_Garage_Cars      1940 non-null   float64
     62  v_Garage_Area      1940 non-null   float64
     63  v_Garage_Qual      1834 non-null   object 
     64  v_Garage_Cond      1834 non-null   object 
     65  v_Paved_Drive      1941 non-null   object 
     66  v_Wood_Deck_SF     1941 non-null   int64  
     67  v_Open_Porch_SF    1941 non-null   int64  
     68  v_Enclosed_Porch   1941 non-null   int64  
     69  v_3Ssn_Porch       1941 non-null   int64  
     70  v_Screen_Porch     1941 non-null   int64  
     71  v_Pool_Area        1941 non-null   int64  
     72  v_Pool_QC          13 non-null     object 
     73  v_Fence            365 non-null    object 
     74  v_Misc_Feature     63 non-null     object 
     75  v_Misc_Val         1941 non-null   int64  
     76  v_Mo_Sold          1941 non-null   int64  
     77  v_Yr_Sold          1941 non-null   int64  
     78  v_Sale_Type        1941 non-null   object 
     79  v_Sale_Condition   1941 non-null   object 
     80  v_SalePrice        1941 non-null   int64  
    dtypes: float64(11), int64(26), object(44)
    memory usage: 1.2+ MB
    Info: None 
    ---
    140000    26
    135000    23
    145000    21
    130000    21
    155000    18
    120000    16
    170000    15
    250000    14
    160000    14
    127000    14
    Name: v_SalePrice, dtype: int64 
    ---
    820 
    ---



```python
#Missing Values

((ht.isna().sum(axis=0)/len(ht)*100) 
    .sort_values(ascending=False)[:13]
    .to_frame(name='% missing') 
    .style.format("{:.1f}"))
```




<style type="text/css">
</style>
<table id="T_6b4f0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6b4f0_level0_col0" class="col_heading level0 col0" >% missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6b4f0_level0_row0" class="row_heading level0 row0" >v_Pool_QC</th>
      <td id="T_6b4f0_row0_col0" class="data row0 col0" >99.3</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row1" class="row_heading level0 row1" >v_Misc_Feature</th>
      <td id="T_6b4f0_row1_col0" class="data row1 col0" >96.8</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row2" class="row_heading level0 row2" >v_Alley</th>
      <td id="T_6b4f0_row2_col0" class="data row2 col0" >93.0</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row3" class="row_heading level0 row3" >v_Fence</th>
      <td id="T_6b4f0_row3_col0" class="data row3 col0" >81.2</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row4" class="row_heading level0 row4" >v_Fireplace_Qu</th>
      <td id="T_6b4f0_row4_col0" class="data row4 col0" >48.4</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row5" class="row_heading level0 row5" >v_Lot_Frontage</th>
      <td id="T_6b4f0_row5_col0" class="data row5 col0" >16.5</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row6" class="row_heading level0 row6" >v_Garage_Cond</th>
      <td id="T_6b4f0_row6_col0" class="data row6 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row7" class="row_heading level0 row7" >v_Garage_Finish</th>
      <td id="T_6b4f0_row7_col0" class="data row7 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row8" class="row_heading level0 row8" >v_Garage_Yr_Blt</th>
      <td id="T_6b4f0_row8_col0" class="data row8 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row9" class="row_heading level0 row9" >v_Garage_Qual</th>
      <td id="T_6b4f0_row9_col0" class="data row9 col0" >5.5</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row10" class="row_heading level0 row10" >v_Garage_Type</th>
      <td id="T_6b4f0_row10_col0" class="data row10 col0" >5.4</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row11" class="row_heading level0 row11" >v_Bsmt_Exposure</th>
      <td id="T_6b4f0_row11_col0" class="data row11 col0" >2.7</td>
    </tr>
    <tr>
      <th id="T_6b4f0_level0_row12" class="row_heading level0 row12" >v_Bsmt_Qual</th>
      <td id="T_6b4f0_row12_col0" class="data row12 col0" >2.6</td>
    </tr>
  </tbody>
</table>





```python
# Outliers
ht.describe(percentiles=[.01,.05,.95,.99]).T.style.format('{:,.2f}')
```




<style type="text/css">
</style>
<table id="T_2e4bc">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2e4bc_level0_col0" class="col_heading level0 col0" >count</th>
      <th id="T_2e4bc_level0_col1" class="col_heading level0 col1" >mean</th>
      <th id="T_2e4bc_level0_col2" class="col_heading level0 col2" >std</th>
      <th id="T_2e4bc_level0_col3" class="col_heading level0 col3" >min</th>
      <th id="T_2e4bc_level0_col4" class="col_heading level0 col4" >1%</th>
      <th id="T_2e4bc_level0_col5" class="col_heading level0 col5" >5%</th>
      <th id="T_2e4bc_level0_col6" class="col_heading level0 col6" >50%</th>
      <th id="T_2e4bc_level0_col7" class="col_heading level0 col7" >95%</th>
      <th id="T_2e4bc_level0_col8" class="col_heading level0 col8" >99%</th>
      <th id="T_2e4bc_level0_col9" class="col_heading level0 col9" >max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2e4bc_level0_row0" class="row_heading level0 row0" >v_MS_SubClass</th>
      <td id="T_2e4bc_row0_col0" class="data row0 col0" >1,941.00</td>
      <td id="T_2e4bc_row0_col1" class="data row0 col1" >58.09</td>
      <td id="T_2e4bc_row0_col2" class="data row0 col2" >42.95</td>
      <td id="T_2e4bc_row0_col3" class="data row0 col3" >20.00</td>
      <td id="T_2e4bc_row0_col4" class="data row0 col4" >20.00</td>
      <td id="T_2e4bc_row0_col5" class="data row0 col5" >20.00</td>
      <td id="T_2e4bc_row0_col6" class="data row0 col6" >50.00</td>
      <td id="T_2e4bc_row0_col7" class="data row0 col7" >160.00</td>
      <td id="T_2e4bc_row0_col8" class="data row0 col8" >190.00</td>
      <td id="T_2e4bc_row0_col9" class="data row0 col9" >190.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row1" class="row_heading level0 row1" >v_Lot_Frontage</th>
      <td id="T_2e4bc_row1_col0" class="data row1 col0" >1,620.00</td>
      <td id="T_2e4bc_row1_col1" class="data row1 col1" >69.30</td>
      <td id="T_2e4bc_row1_col2" class="data row1 col2" >23.98</td>
      <td id="T_2e4bc_row1_col3" class="data row1 col3" >21.00</td>
      <td id="T_2e4bc_row1_col4" class="data row1 col4" >21.00</td>
      <td id="T_2e4bc_row1_col5" class="data row1 col5" >34.00</td>
      <td id="T_2e4bc_row1_col6" class="data row1 col6" >68.00</td>
      <td id="T_2e4bc_row1_col7" class="data row1 col7" >107.05</td>
      <td id="T_2e4bc_row1_col8" class="data row1 col8" >135.81</td>
      <td id="T_2e4bc_row1_col9" class="data row1 col9" >313.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row2" class="row_heading level0 row2" >v_Lot_Area</th>
      <td id="T_2e4bc_row2_col0" class="data row2 col0" >1,941.00</td>
      <td id="T_2e4bc_row2_col1" class="data row2 col1" >10,284.77</td>
      <td id="T_2e4bc_row2_col2" class="data row2 col2" >7,832.30</td>
      <td id="T_2e4bc_row2_col3" class="data row2 col3" >1,470.00</td>
      <td id="T_2e4bc_row2_col4" class="data row2 col4" >1,688.00</td>
      <td id="T_2e4bc_row2_col5" class="data row2 col5" >3,523.00</td>
      <td id="T_2e4bc_row2_col6" class="data row2 col6" >9,450.00</td>
      <td id="T_2e4bc_row2_col7" class="data row2 col7" >17,778.00</td>
      <td id="T_2e4bc_row2_col8" class="data row2 col8" >38,062.40</td>
      <td id="T_2e4bc_row2_col9" class="data row2 col9" >164,660.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row3" class="row_heading level0 row3" >v_Overall_Qual</th>
      <td id="T_2e4bc_row3_col0" class="data row3 col0" >1,941.00</td>
      <td id="T_2e4bc_row3_col1" class="data row3 col1" >6.11</td>
      <td id="T_2e4bc_row3_col2" class="data row3 col2" >1.40</td>
      <td id="T_2e4bc_row3_col3" class="data row3 col3" >1.00</td>
      <td id="T_2e4bc_row3_col4" class="data row3 col4" >3.00</td>
      <td id="T_2e4bc_row3_col5" class="data row3 col5" >4.00</td>
      <td id="T_2e4bc_row3_col6" class="data row3 col6" >6.00</td>
      <td id="T_2e4bc_row3_col7" class="data row3 col7" >8.00</td>
      <td id="T_2e4bc_row3_col8" class="data row3 col8" >10.00</td>
      <td id="T_2e4bc_row3_col9" class="data row3 col9" >10.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row4" class="row_heading level0 row4" >v_Overall_Cond</th>
      <td id="T_2e4bc_row4_col0" class="data row4 col0" >1,941.00</td>
      <td id="T_2e4bc_row4_col1" class="data row4 col1" >5.57</td>
      <td id="T_2e4bc_row4_col2" class="data row4 col2" >1.09</td>
      <td id="T_2e4bc_row4_col3" class="data row4 col3" >1.00</td>
      <td id="T_2e4bc_row4_col4" class="data row4 col4" >3.00</td>
      <td id="T_2e4bc_row4_col5" class="data row4 col5" >4.00</td>
      <td id="T_2e4bc_row4_col6" class="data row4 col6" >5.00</td>
      <td id="T_2e4bc_row4_col7" class="data row4 col7" >8.00</td>
      <td id="T_2e4bc_row4_col8" class="data row4 col8" >9.00</td>
      <td id="T_2e4bc_row4_col9" class="data row4 col9" >9.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row5" class="row_heading level0 row5" >v_Year_Built</th>
      <td id="T_2e4bc_row5_col0" class="data row5 col0" >1,941.00</td>
      <td id="T_2e4bc_row5_col1" class="data row5 col1" >1,971.32</td>
      <td id="T_2e4bc_row5_col2" class="data row5 col2" >30.21</td>
      <td id="T_2e4bc_row5_col3" class="data row5 col3" >1,872.00</td>
      <td id="T_2e4bc_row5_col4" class="data row5 col4" >1,900.00</td>
      <td id="T_2e4bc_row5_col5" class="data row5 col5" >1,916.00</td>
      <td id="T_2e4bc_row5_col6" class="data row5 col6" >1,973.00</td>
      <td id="T_2e4bc_row5_col7" class="data row5 col7" >2,006.00</td>
      <td id="T_2e4bc_row5_col8" class="data row5 col8" >2,007.00</td>
      <td id="T_2e4bc_row5_col9" class="data row5 col9" >2,008.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row6" class="row_heading level0 row6" >v_Year_Remod/Add</th>
      <td id="T_2e4bc_row6_col0" class="data row6 col0" >1,941.00</td>
      <td id="T_2e4bc_row6_col1" class="data row6 col1" >1,984.07</td>
      <td id="T_2e4bc_row6_col2" class="data row6 col2" >20.84</td>
      <td id="T_2e4bc_row6_col3" class="data row6 col3" >1,950.00</td>
      <td id="T_2e4bc_row6_col4" class="data row6 col4" >1,950.00</td>
      <td id="T_2e4bc_row6_col5" class="data row6 col5" >1,950.00</td>
      <td id="T_2e4bc_row6_col6" class="data row6 col6" >1,993.00</td>
      <td id="T_2e4bc_row6_col7" class="data row6 col7" >2,007.00</td>
      <td id="T_2e4bc_row6_col8" class="data row6 col8" >2,008.00</td>
      <td id="T_2e4bc_row6_col9" class="data row6 col9" >2,009.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row7" class="row_heading level0 row7" >v_Mas_Vnr_Area</th>
      <td id="T_2e4bc_row7_col0" class="data row7 col0" >1,923.00</td>
      <td id="T_2e4bc_row7_col1" class="data row7 col1" >104.85</td>
      <td id="T_2e4bc_row7_col2" class="data row7 col2" >184.98</td>
      <td id="T_2e4bc_row7_col3" class="data row7 col3" >0.00</td>
      <td id="T_2e4bc_row7_col4" class="data row7 col4" >0.00</td>
      <td id="T_2e4bc_row7_col5" class="data row7 col5" >0.00</td>
      <td id="T_2e4bc_row7_col6" class="data row7 col6" >0.00</td>
      <td id="T_2e4bc_row7_col7" class="data row7 col7" >472.90</td>
      <td id="T_2e4bc_row7_col8" class="data row7 col8" >794.24</td>
      <td id="T_2e4bc_row7_col9" class="data row7 col9" >1,600.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row8" class="row_heading level0 row8" >v_BsmtFin_SF_1</th>
      <td id="T_2e4bc_row8_col0" class="data row8 col0" >1,940.00</td>
      <td id="T_2e4bc_row8_col1" class="data row8 col1" >436.99</td>
      <td id="T_2e4bc_row8_col2" class="data row8 col2" >457.82</td>
      <td id="T_2e4bc_row8_col3" class="data row8 col3" >0.00</td>
      <td id="T_2e4bc_row8_col4" class="data row8 col4" >0.00</td>
      <td id="T_2e4bc_row8_col5" class="data row8 col5" >0.00</td>
      <td id="T_2e4bc_row8_col6" class="data row8 col6" >361.50</td>
      <td id="T_2e4bc_row8_col7" class="data row8 col7" >1,249.00</td>
      <td id="T_2e4bc_row8_col8" class="data row8 col8" >1,600.93</td>
      <td id="T_2e4bc_row8_col9" class="data row8 col9" >5,644.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row9" class="row_heading level0 row9" >v_BsmtFin_SF_2</th>
      <td id="T_2e4bc_row9_col0" class="data row9 col0" >1,940.00</td>
      <td id="T_2e4bc_row9_col1" class="data row9 col1" >49.25</td>
      <td id="T_2e4bc_row9_col2" class="data row9 col2" >169.56</td>
      <td id="T_2e4bc_row9_col3" class="data row9 col3" >0.00</td>
      <td id="T_2e4bc_row9_col4" class="data row9 col4" >0.00</td>
      <td id="T_2e4bc_row9_col5" class="data row9 col5" >0.00</td>
      <td id="T_2e4bc_row9_col6" class="data row9 col6" >0.00</td>
      <td id="T_2e4bc_row9_col7" class="data row9 col7" >435.00</td>
      <td id="T_2e4bc_row9_col8" class="data row9 col8" >883.98</td>
      <td id="T_2e4bc_row9_col9" class="data row9 col9" >1,474.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row10" class="row_heading level0 row10" >v_Bsmt_Unf_SF</th>
      <td id="T_2e4bc_row10_col0" class="data row10 col0" >1,940.00</td>
      <td id="T_2e4bc_row10_col1" class="data row10 col1" >567.44</td>
      <td id="T_2e4bc_row10_col2" class="data row10 col2" >439.60</td>
      <td id="T_2e4bc_row10_col3" class="data row10 col3" >0.00</td>
      <td id="T_2e4bc_row10_col4" class="data row10 col4" >0.00</td>
      <td id="T_2e4bc_row10_col5" class="data row10 col5" >0.00</td>
      <td id="T_2e4bc_row10_col6" class="data row10 col6" >474.00</td>
      <td id="T_2e4bc_row10_col7" class="data row10 col7" >1,488.05</td>
      <td id="T_2e4bc_row10_col8" class="data row10 col8" >1,775.83</td>
      <td id="T_2e4bc_row10_col9" class="data row10 col9" >2,153.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row11" class="row_heading level0 row11" >v_Total_Bsmt_SF</th>
      <td id="T_2e4bc_row11_col0" class="data row11 col0" >1,940.00</td>
      <td id="T_2e4bc_row11_col1" class="data row11 col1" >1,053.67</td>
      <td id="T_2e4bc_row11_col2" class="data row11 col2" >438.66</td>
      <td id="T_2e4bc_row11_col3" class="data row11 col3" >0.00</td>
      <td id="T_2e4bc_row11_col4" class="data row11 col4" >0.00</td>
      <td id="T_2e4bc_row11_col5" class="data row11 col5" >483.00</td>
      <td id="T_2e4bc_row11_col6" class="data row11 col6" >989.50</td>
      <td id="T_2e4bc_row11_col7" class="data row11 col7" >1,776.05</td>
      <td id="T_2e4bc_row11_col8" class="data row11 col8" >2,138.44</td>
      <td id="T_2e4bc_row11_col9" class="data row11 col9" >6,110.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row12" class="row_heading level0 row12" >v_1st_Flr_SF</th>
      <td id="T_2e4bc_row12_col0" class="data row12 col0" >1,941.00</td>
      <td id="T_2e4bc_row12_col1" class="data row12 col1" >1,161.07</td>
      <td id="T_2e4bc_row12_col2" class="data row12 col2" >396.95</td>
      <td id="T_2e4bc_row12_col3" class="data row12 col3" >334.00</td>
      <td id="T_2e4bc_row12_col4" class="data row12 col4" >507.60</td>
      <td id="T_2e4bc_row12_col5" class="data row12 col5" >672.00</td>
      <td id="T_2e4bc_row12_col6" class="data row12 col6" >1,085.00</td>
      <td id="T_2e4bc_row12_col7" class="data row12 col7" >1,828.00</td>
      <td id="T_2e4bc_row12_col8" class="data row12 col8" >2,277.80</td>
      <td id="T_2e4bc_row12_col9" class="data row12 col9" >5,095.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row13" class="row_heading level0 row13" >v_2nd_Flr_SF</th>
      <td id="T_2e4bc_row13_col0" class="data row13 col0" >1,941.00</td>
      <td id="T_2e4bc_row13_col1" class="data row13 col1" >340.96</td>
      <td id="T_2e4bc_row13_col2" class="data row13 col2" >434.24</td>
      <td id="T_2e4bc_row13_col3" class="data row13 col3" >0.00</td>
      <td id="T_2e4bc_row13_col4" class="data row13 col4" >0.00</td>
      <td id="T_2e4bc_row13_col5" class="data row13 col5" >0.00</td>
      <td id="T_2e4bc_row13_col6" class="data row13 col6" >0.00</td>
      <td id="T_2e4bc_row13_col7" class="data row13 col7" >1,142.00</td>
      <td id="T_2e4bc_row13_col8" class="data row13 col8" >1,406.20</td>
      <td id="T_2e4bc_row13_col9" class="data row13 col9" >2,065.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row14" class="row_heading level0 row14" >v_Low_Qual_Fin_SF</th>
      <td id="T_2e4bc_row14_col0" class="data row14 col0" >1,941.00</td>
      <td id="T_2e4bc_row14_col1" class="data row14 col1" >4.28</td>
      <td id="T_2e4bc_row14_col2" class="data row14 col2" >42.94</td>
      <td id="T_2e4bc_row14_col3" class="data row14 col3" >0.00</td>
      <td id="T_2e4bc_row14_col4" class="data row14 col4" >0.00</td>
      <td id="T_2e4bc_row14_col5" class="data row14 col5" >0.00</td>
      <td id="T_2e4bc_row14_col6" class="data row14 col6" >0.00</td>
      <td id="T_2e4bc_row14_col7" class="data row14 col7" >0.00</td>
      <td id="T_2e4bc_row14_col8" class="data row14 col8" >111.60</td>
      <td id="T_2e4bc_row14_col9" class="data row14 col9" >697.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row15" class="row_heading level0 row15" >v_Gr_Liv_Area</th>
      <td id="T_2e4bc_row15_col0" class="data row15 col0" >1,941.00</td>
      <td id="T_2e4bc_row15_col1" class="data row15 col1" >1,506.31</td>
      <td id="T_2e4bc_row15_col2" class="data row15 col2" >524.77</td>
      <td id="T_2e4bc_row15_col3" class="data row15 col3" >334.00</td>
      <td id="T_2e4bc_row15_col4" class="data row15 col4" >691.80</td>
      <td id="T_2e4bc_row15_col5" class="data row15 col5" >864.00</td>
      <td id="T_2e4bc_row15_col6" class="data row15 col6" >1,436.00</td>
      <td id="T_2e4bc_row15_col7" class="data row15 col7" >2,500.00</td>
      <td id="T_2e4bc_row15_col8" class="data row15 col8" >3,029.20</td>
      <td id="T_2e4bc_row15_col9" class="data row15 col9" >5,642.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row16" class="row_heading level0 row16" >v_Bsmt_Full_Bath</th>
      <td id="T_2e4bc_row16_col0" class="data row16 col0" >1,939.00</td>
      <td id="T_2e4bc_row16_col1" class="data row16 col1" >0.42</td>
      <td id="T_2e4bc_row16_col2" class="data row16 col2" >0.52</td>
      <td id="T_2e4bc_row16_col3" class="data row16 col3" >0.00</td>
      <td id="T_2e4bc_row16_col4" class="data row16 col4" >0.00</td>
      <td id="T_2e4bc_row16_col5" class="data row16 col5" >0.00</td>
      <td id="T_2e4bc_row16_col6" class="data row16 col6" >0.00</td>
      <td id="T_2e4bc_row16_col7" class="data row16 col7" >1.00</td>
      <td id="T_2e4bc_row16_col8" class="data row16 col8" >2.00</td>
      <td id="T_2e4bc_row16_col9" class="data row16 col9" >2.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row17" class="row_heading level0 row17" >v_Bsmt_Half_Bath</th>
      <td id="T_2e4bc_row17_col0" class="data row17 col0" >1,939.00</td>
      <td id="T_2e4bc_row17_col1" class="data row17 col1" >0.06</td>
      <td id="T_2e4bc_row17_col2" class="data row17 col2" >0.25</td>
      <td id="T_2e4bc_row17_col3" class="data row17 col3" >0.00</td>
      <td id="T_2e4bc_row17_col4" class="data row17 col4" >0.00</td>
      <td id="T_2e4bc_row17_col5" class="data row17 col5" >0.00</td>
      <td id="T_2e4bc_row17_col6" class="data row17 col6" >0.00</td>
      <td id="T_2e4bc_row17_col7" class="data row17 col7" >1.00</td>
      <td id="T_2e4bc_row17_col8" class="data row17 col8" >1.00</td>
      <td id="T_2e4bc_row17_col9" class="data row17 col9" >2.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row18" class="row_heading level0 row18" >v_Full_Bath</th>
      <td id="T_2e4bc_row18_col0" class="data row18 col0" >1,941.00</td>
      <td id="T_2e4bc_row18_col1" class="data row18 col1" >1.57</td>
      <td id="T_2e4bc_row18_col2" class="data row18 col2" >0.55</td>
      <td id="T_2e4bc_row18_col3" class="data row18 col3" >0.00</td>
      <td id="T_2e4bc_row18_col4" class="data row18 col4" >1.00</td>
      <td id="T_2e4bc_row18_col5" class="data row18 col5" >1.00</td>
      <td id="T_2e4bc_row18_col6" class="data row18 col6" >2.00</td>
      <td id="T_2e4bc_row18_col7" class="data row18 col7" >2.00</td>
      <td id="T_2e4bc_row18_col8" class="data row18 col8" >3.00</td>
      <td id="T_2e4bc_row18_col9" class="data row18 col9" >3.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row19" class="row_heading level0 row19" >v_Half_Bath</th>
      <td id="T_2e4bc_row19_col0" class="data row19 col0" >1,941.00</td>
      <td id="T_2e4bc_row19_col1" class="data row19 col1" >0.38</td>
      <td id="T_2e4bc_row19_col2" class="data row19 col2" >0.50</td>
      <td id="T_2e4bc_row19_col3" class="data row19 col3" >0.00</td>
      <td id="T_2e4bc_row19_col4" class="data row19 col4" >0.00</td>
      <td id="T_2e4bc_row19_col5" class="data row19 col5" >0.00</td>
      <td id="T_2e4bc_row19_col6" class="data row19 col6" >0.00</td>
      <td id="T_2e4bc_row19_col7" class="data row19 col7" >1.00</td>
      <td id="T_2e4bc_row19_col8" class="data row19 col8" >1.00</td>
      <td id="T_2e4bc_row19_col9" class="data row19 col9" >2.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row20" class="row_heading level0 row20" >v_Bedroom_AbvGr</th>
      <td id="T_2e4bc_row20_col0" class="data row20 col0" >1,941.00</td>
      <td id="T_2e4bc_row20_col1" class="data row20 col1" >2.87</td>
      <td id="T_2e4bc_row20_col2" class="data row20 col2" >0.83</td>
      <td id="T_2e4bc_row20_col3" class="data row20 col3" >0.00</td>
      <td id="T_2e4bc_row20_col4" class="data row20 col4" >1.00</td>
      <td id="T_2e4bc_row20_col5" class="data row20 col5" >2.00</td>
      <td id="T_2e4bc_row20_col6" class="data row20 col6" >3.00</td>
      <td id="T_2e4bc_row20_col7" class="data row20 col7" >4.00</td>
      <td id="T_2e4bc_row20_col8" class="data row20 col8" >5.00</td>
      <td id="T_2e4bc_row20_col9" class="data row20 col9" >8.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row21" class="row_heading level0 row21" >v_Kitchen_AbvGr</th>
      <td id="T_2e4bc_row21_col0" class="data row21 col0" >1,941.00</td>
      <td id="T_2e4bc_row21_col1" class="data row21 col1" >1.04</td>
      <td id="T_2e4bc_row21_col2" class="data row21 col2" >0.20</td>
      <td id="T_2e4bc_row21_col3" class="data row21 col3" >0.00</td>
      <td id="T_2e4bc_row21_col4" class="data row21 col4" >1.00</td>
      <td id="T_2e4bc_row21_col5" class="data row21 col5" >1.00</td>
      <td id="T_2e4bc_row21_col6" class="data row21 col6" >1.00</td>
      <td id="T_2e4bc_row21_col7" class="data row21 col7" >1.00</td>
      <td id="T_2e4bc_row21_col8" class="data row21 col8" >2.00</td>
      <td id="T_2e4bc_row21_col9" class="data row21 col9" >2.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row22" class="row_heading level0 row22" >v_TotRms_AbvGrd</th>
      <td id="T_2e4bc_row22_col0" class="data row22 col0" >1,941.00</td>
      <td id="T_2e4bc_row22_col1" class="data row22 col1" >6.47</td>
      <td id="T_2e4bc_row22_col2" class="data row22 col2" >1.58</td>
      <td id="T_2e4bc_row22_col3" class="data row22 col3" >2.00</td>
      <td id="T_2e4bc_row22_col4" class="data row22 col4" >4.00</td>
      <td id="T_2e4bc_row22_col5" class="data row22 col5" >4.00</td>
      <td id="T_2e4bc_row22_col6" class="data row22 col6" >6.00</td>
      <td id="T_2e4bc_row22_col7" class="data row22 col7" >9.00</td>
      <td id="T_2e4bc_row22_col8" class="data row22 col8" >11.00</td>
      <td id="T_2e4bc_row22_col9" class="data row22 col9" >15.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row23" class="row_heading level0 row23" >v_Fireplaces</th>
      <td id="T_2e4bc_row23_col0" class="data row23 col0" >1,941.00</td>
      <td id="T_2e4bc_row23_col1" class="data row23 col1" >0.60</td>
      <td id="T_2e4bc_row23_col2" class="data row23 col2" >0.64</td>
      <td id="T_2e4bc_row23_col3" class="data row23 col3" >0.00</td>
      <td id="T_2e4bc_row23_col4" class="data row23 col4" >0.00</td>
      <td id="T_2e4bc_row23_col5" class="data row23 col5" >0.00</td>
      <td id="T_2e4bc_row23_col6" class="data row23 col6" >1.00</td>
      <td id="T_2e4bc_row23_col7" class="data row23 col7" >2.00</td>
      <td id="T_2e4bc_row23_col8" class="data row23 col8" >2.00</td>
      <td id="T_2e4bc_row23_col9" class="data row23 col9" >4.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row24" class="row_heading level0 row24" >v_Garage_Yr_Blt</th>
      <td id="T_2e4bc_row24_col0" class="data row24 col0" >1,834.00</td>
      <td id="T_2e4bc_row24_col1" class="data row24 col1" >1,978.19</td>
      <td id="T_2e4bc_row24_col2" class="data row24 col2" >25.73</td>
      <td id="T_2e4bc_row24_col3" class="data row24 col3" >1,895.00</td>
      <td id="T_2e4bc_row24_col4" class="data row24 col4" >1,916.00</td>
      <td id="T_2e4bc_row24_col5" class="data row24 col5" >1,928.65</td>
      <td id="T_2e4bc_row24_col6" class="data row24 col6" >1,980.00</td>
      <td id="T_2e4bc_row24_col7" class="data row24 col7" >2,007.00</td>
      <td id="T_2e4bc_row24_col8" class="data row24 col8" >2,008.00</td>
      <td id="T_2e4bc_row24_col9" class="data row24 col9" >2,207.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row25" class="row_heading level0 row25" >v_Garage_Cars</th>
      <td id="T_2e4bc_row25_col0" class="data row25 col0" >1,940.00</td>
      <td id="T_2e4bc_row25_col1" class="data row25 col1" >1.77</td>
      <td id="T_2e4bc_row25_col2" class="data row25 col2" >0.76</td>
      <td id="T_2e4bc_row25_col3" class="data row25 col3" >0.00</td>
      <td id="T_2e4bc_row25_col4" class="data row25 col4" >0.00</td>
      <td id="T_2e4bc_row25_col5" class="data row25 col5" >0.00</td>
      <td id="T_2e4bc_row25_col6" class="data row25 col6" >2.00</td>
      <td id="T_2e4bc_row25_col7" class="data row25 col7" >3.00</td>
      <td id="T_2e4bc_row25_col8" class="data row25 col8" >3.00</td>
      <td id="T_2e4bc_row25_col9" class="data row25 col9" >4.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row26" class="row_heading level0 row26" >v_Garage_Area</th>
      <td id="T_2e4bc_row26_col0" class="data row26 col0" >1,940.00</td>
      <td id="T_2e4bc_row26_col1" class="data row26 col1" >472.77</td>
      <td id="T_2e4bc_row26_col2" class="data row26 col2" >217.09</td>
      <td id="T_2e4bc_row26_col3" class="data row26 col3" >0.00</td>
      <td id="T_2e4bc_row26_col4" class="data row26 col4" >0.00</td>
      <td id="T_2e4bc_row26_col5" class="data row26 col5" >0.00</td>
      <td id="T_2e4bc_row26_col6" class="data row26 col6" >478.00</td>
      <td id="T_2e4bc_row26_col7" class="data row26 col7" >859.25</td>
      <td id="T_2e4bc_row26_col8" class="data row26 col8" >1,040.61</td>
      <td id="T_2e4bc_row26_col9" class="data row26 col9" >1,488.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row27" class="row_heading level0 row27" >v_Wood_Deck_SF</th>
      <td id="T_2e4bc_row27_col0" class="data row27 col0" >1,941.00</td>
      <td id="T_2e4bc_row27_col1" class="data row27 col1" >92.46</td>
      <td id="T_2e4bc_row27_col2" class="data row27 col2" >127.02</td>
      <td id="T_2e4bc_row27_col3" class="data row27 col3" >0.00</td>
      <td id="T_2e4bc_row27_col4" class="data row27 col4" >0.00</td>
      <td id="T_2e4bc_row27_col5" class="data row27 col5" >0.00</td>
      <td id="T_2e4bc_row27_col6" class="data row27 col6" >0.00</td>
      <td id="T_2e4bc_row27_col7" class="data row27 col7" >320.00</td>
      <td id="T_2e4bc_row27_col8" class="data row27 col8" >515.80</td>
      <td id="T_2e4bc_row27_col9" class="data row27 col9" >1,424.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row28" class="row_heading level0 row28" >v_Open_Porch_SF</th>
      <td id="T_2e4bc_row28_col0" class="data row28 col0" >1,941.00</td>
      <td id="T_2e4bc_row28_col1" class="data row28 col1" >49.16</td>
      <td id="T_2e4bc_row28_col2" class="data row28 col2" >70.30</td>
      <td id="T_2e4bc_row28_col3" class="data row28 col3" >0.00</td>
      <td id="T_2e4bc_row28_col4" class="data row28 col4" >0.00</td>
      <td id="T_2e4bc_row28_col5" class="data row28 col5" >0.00</td>
      <td id="T_2e4bc_row28_col6" class="data row28 col6" >28.00</td>
      <td id="T_2e4bc_row28_col7" class="data row28 col7" >189.00</td>
      <td id="T_2e4bc_row28_col8" class="data row28 col8" >296.20</td>
      <td id="T_2e4bc_row28_col9" class="data row28 col9" >742.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row29" class="row_heading level0 row29" >v_Enclosed_Porch</th>
      <td id="T_2e4bc_row29_col0" class="data row29 col0" >1,941.00</td>
      <td id="T_2e4bc_row29_col1" class="data row29 col1" >22.95</td>
      <td id="T_2e4bc_row29_col2" class="data row29 col2" >65.25</td>
      <td id="T_2e4bc_row29_col3" class="data row29 col3" >0.00</td>
      <td id="T_2e4bc_row29_col4" class="data row29 col4" >0.00</td>
      <td id="T_2e4bc_row29_col5" class="data row29 col5" >0.00</td>
      <td id="T_2e4bc_row29_col6" class="data row29 col6" >0.00</td>
      <td id="T_2e4bc_row29_col7" class="data row29 col7" >180.00</td>
      <td id="T_2e4bc_row29_col8" class="data row29 col8" >262.00</td>
      <td id="T_2e4bc_row29_col9" class="data row29 col9" >1,012.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row30" class="row_heading level0 row30" >v_3Ssn_Porch</th>
      <td id="T_2e4bc_row30_col0" class="data row30 col0" >1,941.00</td>
      <td id="T_2e4bc_row30_col1" class="data row30 col1" >2.25</td>
      <td id="T_2e4bc_row30_col2" class="data row30 col2" >22.42</td>
      <td id="T_2e4bc_row30_col3" class="data row30 col3" >0.00</td>
      <td id="T_2e4bc_row30_col4" class="data row30 col4" >0.00</td>
      <td id="T_2e4bc_row30_col5" class="data row30 col5" >0.00</td>
      <td id="T_2e4bc_row30_col6" class="data row30 col6" >0.00</td>
      <td id="T_2e4bc_row30_col7" class="data row30 col7" >0.00</td>
      <td id="T_2e4bc_row30_col8" class="data row30 col8" >110.40</td>
      <td id="T_2e4bc_row30_col9" class="data row30 col9" >407.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row31" class="row_heading level0 row31" >v_Screen_Porch</th>
      <td id="T_2e4bc_row31_col0" class="data row31 col0" >1,941.00</td>
      <td id="T_2e4bc_row31_col1" class="data row31 col1" >16.25</td>
      <td id="T_2e4bc_row31_col2" class="data row31 col2" >56.75</td>
      <td id="T_2e4bc_row31_col3" class="data row31 col3" >0.00</td>
      <td id="T_2e4bc_row31_col4" class="data row31 col4" >0.00</td>
      <td id="T_2e4bc_row31_col5" class="data row31 col5" >0.00</td>
      <td id="T_2e4bc_row31_col6" class="data row31 col6" >0.00</td>
      <td id="T_2e4bc_row31_col7" class="data row31 col7" >162.00</td>
      <td id="T_2e4bc_row31_col8" class="data row31 col8" >263.60</td>
      <td id="T_2e4bc_row31_col9" class="data row31 col9" >576.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row32" class="row_heading level0 row32" >v_Pool_Area</th>
      <td id="T_2e4bc_row32_col0" class="data row32 col0" >1,941.00</td>
      <td id="T_2e4bc_row32_col1" class="data row32 col1" >3.39</td>
      <td id="T_2e4bc_row32_col2" class="data row32 col2" >43.70</td>
      <td id="T_2e4bc_row32_col3" class="data row32 col3" >0.00</td>
      <td id="T_2e4bc_row32_col4" class="data row32 col4" >0.00</td>
      <td id="T_2e4bc_row32_col5" class="data row32 col5" >0.00</td>
      <td id="T_2e4bc_row32_col6" class="data row32 col6" >0.00</td>
      <td id="T_2e4bc_row32_col7" class="data row32 col7" >0.00</td>
      <td id="T_2e4bc_row32_col8" class="data row32 col8" >0.00</td>
      <td id="T_2e4bc_row32_col9" class="data row32 col9" >800.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row33" class="row_heading level0 row33" >v_Misc_Val</th>
      <td id="T_2e4bc_row33_col0" class="data row33 col0" >1,941.00</td>
      <td id="T_2e4bc_row33_col1" class="data row33 col1" >52.55</td>
      <td id="T_2e4bc_row33_col2" class="data row33 col2" >616.06</td>
      <td id="T_2e4bc_row33_col3" class="data row33 col3" >0.00</td>
      <td id="T_2e4bc_row33_col4" class="data row33 col4" >0.00</td>
      <td id="T_2e4bc_row33_col5" class="data row33 col5" >0.00</td>
      <td id="T_2e4bc_row33_col6" class="data row33 col6" >0.00</td>
      <td id="T_2e4bc_row33_col7" class="data row33 col7" >0.00</td>
      <td id="T_2e4bc_row33_col8" class="data row33 col8" >900.00</td>
      <td id="T_2e4bc_row33_col9" class="data row33 col9" >17,000.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row34" class="row_heading level0 row34" >v_Mo_Sold</th>
      <td id="T_2e4bc_row34_col0" class="data row34 col0" >1,941.00</td>
      <td id="T_2e4bc_row34_col1" class="data row34 col1" >6.43</td>
      <td id="T_2e4bc_row34_col2" class="data row34 col2" >2.75</td>
      <td id="T_2e4bc_row34_col3" class="data row34 col3" >1.00</td>
      <td id="T_2e4bc_row34_col4" class="data row34 col4" >1.00</td>
      <td id="T_2e4bc_row34_col5" class="data row34 col5" >2.00</td>
      <td id="T_2e4bc_row34_col6" class="data row34 col6" >6.00</td>
      <td id="T_2e4bc_row34_col7" class="data row34 col7" >11.00</td>
      <td id="T_2e4bc_row34_col8" class="data row34 col8" >12.00</td>
      <td id="T_2e4bc_row34_col9" class="data row34 col9" >12.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row35" class="row_heading level0 row35" >v_Yr_Sold</th>
      <td id="T_2e4bc_row35_col0" class="data row35 col0" >1,941.00</td>
      <td id="T_2e4bc_row35_col1" class="data row35 col1" >2,007.00</td>
      <td id="T_2e4bc_row35_col2" class="data row35 col2" >0.80</td>
      <td id="T_2e4bc_row35_col3" class="data row35 col3" >2,006.00</td>
      <td id="T_2e4bc_row35_col4" class="data row35 col4" >2,006.00</td>
      <td id="T_2e4bc_row35_col5" class="data row35 col5" >2,006.00</td>
      <td id="T_2e4bc_row35_col6" class="data row35 col6" >2,007.00</td>
      <td id="T_2e4bc_row35_col7" class="data row35 col7" >2,008.00</td>
      <td id="T_2e4bc_row35_col8" class="data row35 col8" >2,008.00</td>
      <td id="T_2e4bc_row35_col9" class="data row35 col9" >2,008.00</td>
    </tr>
    <tr>
      <th id="T_2e4bc_level0_row36" class="row_heading level0 row36" >v_SalePrice</th>
      <td id="T_2e4bc_row36_col0" class="data row36 col0" >1,941.00</td>
      <td id="T_2e4bc_row36_col1" class="data row36 col1" >182,033.24</td>
      <td id="T_2e4bc_row36_col2" class="data row36 col2" >80,407.10</td>
      <td id="T_2e4bc_row36_col3" class="data row36 col3" >13,100.00</td>
      <td id="T_2e4bc_row36_col4" class="data row36 col4" >64,700.00</td>
      <td id="T_2e4bc_row36_col5" class="data row36 col5" >89,500.00</td>
      <td id="T_2e4bc_row36_col6" class="data row36 col6" >161,900.00</td>
      <td id="T_2e4bc_row36_col7" class="data row36 col7" >339,750.00</td>
      <td id="T_2e4bc_row36_col8" class="data row36 col8" >453,000.00</td>
      <td id="T_2e4bc_row36_col9" class="data row36 col9" >755,000.00</td>
    </tr>
  </tbody>
</table>




## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? I won't be shocked if someone beats that!
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
ht.columns
```




    Index(['parcel', 'v_MS_SubClass', 'v_MS_Zoning', 'v_Lot_Frontage',
           'v_Lot_Area', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour',
           'v_Utilities', 'v_Lot_Config', 'v_Land_Slope', 'v_Neighborhood',
           'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
           'v_Overall_Qual', 'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add',
           'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd',
           'v_Mas_Vnr_Type', 'v_Mas_Vnr_Area', 'v_Exter_Qual', 'v_Exter_Cond',
           'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
           'v_BsmtFin_Type_1', 'v_BsmtFin_SF_1', 'v_BsmtFin_Type_2',
           'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_Heating',
           'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_1st_Flr_SF',
           'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
           'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
           'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_Kitchen_Qual',
           'v_TotRms_AbvGrd', 'v_Functional', 'v_Fireplaces', 'v_Fireplace_Qu',
           'v_Garage_Type', 'v_Garage_Yr_Blt', 'v_Garage_Finish', 'v_Garage_Cars',
           'v_Garage_Area', 'v_Garage_Qual', 'v_Garage_Cond', 'v_Paved_Drive',
           'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch',
           'v_Screen_Porch', 'v_Pool_Area', 'v_Pool_QC', 'v_Fence',
           'v_Misc_Feature', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_Sale_Type',
           'v_Sale_Condition', 'v_SalePrice'],
          dtype='object')




```python
# one var: 'y ~ x' means fit y = a + b*X

reg1 = sm_ols('v_SalePrice ~  v_Lot_Area ', data=ht).fit()

reg2= sm_ols('v_SalePrice ~  np.log(v_Lot_Area)  ',  data=ht).fit()

reg3= sm_ols('np.log(v_SalePrice) ~  v_Lot_Area  ',  data=ht).fit()

reg4= sm_ols('np.log(v_SalePrice) ~  np.log(v_Lot_Area)  ',  data=ht).fit()

reg5= sm_ols('np.log(v_SalePrice) ~  v_Yr_Sold  ',  data=ht).fit()

# multiple variables: just add them to the formula
# 'y ~ x1 + x2' means fit y = a + b*x1 + c*x2
reg6 = sm_ols('np.log(v_SalePrice) ~  v_Yr_Sold==2007 + v_Yr_Sold==2008 ',  data=ht).fit()

reg7 = sm_ols('np.log(v_SalePrice) ~ v_Sale_Condition + v_Neighborhood + v_Year_Built + v_House_Style + v_Foundation',
              data = ht).fit()
```


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}


# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y = interest rate if not specified, log(interest rate else)')
print(summary_col(results=[reg1,reg2,reg3,reg4,reg5,reg6, reg7], # list the result obj here
                  float_format='%0.3f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 ','4 ','5','6', '7'],  #just use the y variable name
                  info_dict=info_dict,
                  #regressor_order=[ 'Intercept','Borrower_Credit_Score_at_Origination','l_credscore','l_LTV','l_credscore:l_LTV',
                               #   'C(creditbins)[Very Poor]','C(creditbins)[Fair]','C(creditbins)[Good]','C(creditbins)[Vrey Good]','C(creditbins)[Exceptional]']
                  ))

```

    ============================================================================================================
                      y = interest rate if not specified, log(interest rate else)
    
    ========================================================================================================
                                      1             2            3        4        5         6         7    
    --------------------------------------------------------------------------------------------------------
    Intercept                   154789.550*** -327915.802*** 11.894*** 9.405*** 22.293   12.023*** 4.565*** 
                                (2911.591)    (30221.347)    (0.015)   (0.151)  (22.937) (0.016)   (0.889)  
    R-squared                   0.067         0.128          0.065     0.135    0.000    0.001     0.664    
    R-squared Adj.              0.066         0.128          0.064     0.135    -0.000   0.000     0.656    
    np.log(v_Lot_Area)                        56028.170***             0.288***                             
                                              (3315.139)               (0.017)                              
    v_Foundation[T.CBlock]                                                                         -0.004   
                                                                                                   (0.024)  
    v_Foundation[T.PConc]                                                                          0.055**  
                                                                                                   (0.027)  
    v_Foundation[T.Slab]                                                                           -0.296***
                                                                                                   (0.050)  
    v_Foundation[T.Stone]                                                                          -0.060   
                                                                                                   (0.099)  
    v_Foundation[T.Wood]                                                                           -0.034   
                                                                                                   (0.173)  
    v_House_Style[T.1.5Unf]                                                                        -0.199***
                                                                                                   (0.062)  
    v_House_Style[T.1Story]                                                                        -0.129***
                                                                                                   (0.021)  
    v_House_Style[T.2.5Fin]                                                                        0.549*** 
                                                                                                   (0.101)  
    v_House_Style[T.2.5Unf]                                                                        0.237*** 
                                                                                                   (0.058)  
    v_House_Style[T.2Story]                                                                        0.001    
                                                                                                   (0.022)  
    v_House_Style[T.SFoyer]                                                                        -0.143***
                                                                                                   (0.040)  
    v_House_Style[T.SLvl]                                                                          -0.081** 
                                                                                                   (0.033)  
    v_Lot_Area                  2.649***                     0.000***                                       
                                (0.225)                      (0.000)                                        
    v_Neighborhood[T.Blueste]                                                                      -0.237*  
                                                                                                   (0.132)  
    v_Neighborhood[T.BrDale]                                                                       -0.551***
                                                                                                   (0.078)  
    v_Neighborhood[T.BrkSide]                                                                      -0.239***
                                                                                                   (0.067)  
    v_Neighborhood[T.ClearCr]                                                                      0.219*** 
                                                                                                   (0.071)  
    v_Neighborhood[T.CollgCr]                                                                      -0.022   
                                                                                                   (0.056)  
    v_Neighborhood[T.Crawfor]                                                                      0.189*** 
                                                                                                   (0.064)  
    v_Neighborhood[T.Edwards]                                                                      -0.219***
                                                                                                   (0.061)  
    v_Neighborhood[T.Gilbert]                                                                      -0.126** 
                                                                                                   (0.059)  
    v_Neighborhood[T.Greens]                                                                       0.168    
                                                                                                   (0.120)  
    v_Neighborhood[T.GrnHill]                                                                      0.439**  
                                                                                                   (0.176)  
    v_Neighborhood[T.IDOTRR]                                                                       -0.365***
                                                                                                   (0.067)  
    v_Neighborhood[T.Landmrk]                                                                      -0.423*  
                                                                                                   (0.243)  
    v_Neighborhood[T.MeadowV]                                                                      -0.582***
                                                                                                   (0.074)  
    v_Neighborhood[T.Mitchel]                                                                      -0.112*  
                                                                                                   (0.062)  
    v_Neighborhood[T.NAmes]                                                                        -0.082   
                                                                                                   (0.059)  
    v_Neighborhood[T.NPkVill]                                                                      -0.162*  
                                                                                                   (0.097)  
    v_Neighborhood[T.NWAmes]                                                                       0.074    
                                                                                                   (0.062)  
    v_Neighborhood[T.NoRidge]                                                                      0.466*** 
                                                                                                   (0.065)  
    v_Neighborhood[T.NridgHt]                                                                      0.365*** 
                                                                                                   (0.058)  
    v_Neighborhood[T.OldTown]                                                                      -0.201***
                                                                                                   (0.065)  
    v_Neighborhood[T.SWISU]                                                                        -0.182** 
                                                                                                   (0.076)  
    v_Neighborhood[T.SawyerW]                                                                      -0.056   
                                                                                                   (0.061)  
    v_Neighborhood[T.Sawyer]                                                                       -0.128** 
                                                                                                   (0.062)  
    v_Neighborhood[T.Somerst]                                                                      0.045    
                                                                                                   (0.058)  
    v_Neighborhood[T.StoneBr]                                                                      0.418*** 
                                                                                                   (0.066)  
    v_Neighborhood[T.Timber]                                                                       0.219*** 
                                                                                                   (0.064)  
    v_Neighborhood[T.Veenker]                                                                      0.310*** 
                                                                                                   (0.076)  
    v_Sale_Condition[T.AdjLand]                                                                    -0.018   
                                                                                                   (0.074)  
    v_Sale_Condition[T.Alloca]                                                                     0.286*** 
                                                                                                   (0.073)  
    v_Sale_Condition[T.Family]                                                                     0.114**  
                                                                                                   (0.046)  
    v_Sale_Condition[T.Normal]                                                                     0.113*** 
                                                                                                   (0.022)  
    v_Sale_Condition[T.Partial]                                                                    0.231*** 
                                                                                                   (0.029)  
    v_Year_Built                                                                                   0.004*** 
                                                                                                   (0.000)  
    v_Yr_Sold                                                                   -0.005                      
                                                                                (0.011)                     
    v_Yr_Sold == 2007[T.True]                                                            0.026              
                                                                                         (0.022)            
    v_Yr_Sold == 2008[T.True]                                                            -0.010             
                                                                                         (0.023)            
    R-squared                   0.07          0.13           0.06      0.13     0.00     0.00      0.66     
    Adj R-squared               0.07          0.13           0.06      0.13     -0.00    0.00      0.66     
    No. observations            1941          1941           1941      1941     1941     1941      1941     
    ========================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 


**#2** When lot area goes up by 1%, the sales price will increase by $560.28 

**#3** When lot area goes up by 1 unit, the sales price will go up by 0.0013%

**#4** I think that the model that best explains the data is model 2 as it has standard error below .01 and the highest R2 of all the models.

**#5** When year sold goes up by 1 year, the sales price will down  by 1%

**#6** 12.02% of sales price when the year is not 2007 or 2008.

**#7** In the year 2007, sales price goes up 3%. In the year 2008, sales price goes down 1%

**#8** The R2 of Model 6 higher than the R2 of Model 5 is becuase as you increase the number of regressors in a model, the R2 will increase. This is because the more varible one adds, the large the increase in the sum of squares becomes which is why R2 increases.

**#9** the vairables I chose to test were Sale Condition, Neighborhood, Year_Built, House Style, and Foundation.

**#10** The R2 of my seventh regression is 0.66.

**#11**  No becuase it's so time focused, the conditions of those specifc years are unliekly to occur so exactly again that it would be a good prediction.

**#12**  Yes, you could examine trends through time in order to make predictions about what the housing market would look like in the future.


```python

```
