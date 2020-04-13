# COVID-19-pop-outflow
## This is the Python project for Nature manuscript 2020-02-01778B  
Python development environment used: **Python 3.7.3**  
**numpy** version required: 1.16.2  
**pandas** version required: 0.24.2  
**matplotlib** version required: 3.2.0  
**lmfit** version required: 1.0.0

## Files description
### nature-data/pneumonia_panel_296_cities(submit).csv 
It's the basic data panel for our research, including 296 prefecture-level cities in China(based on CHINA CITY STATISTICAL YEARBOOK(2018)), and the time window of 27 days(from Jan. 24 to Feb. 19, 2020).   
The ***GDP and population*** data is collected from YEARBOOK, however, there are some missing values of GDP in the YEARBOOK, so we queried the GDP values of these cities in 2017 from the Chinese statistical information website(www.tjcn.org), details as following:  
1)Jiayuguan(嘉峪关): http://www.tjcn.org/tjgb/28gs/35595.html  
2)Danzhou(儋州): http://www.tjcn.org/tjgb/21hn/35463.html  
3)Zhongshan(中山): http://www.tjcn.org/tjgb/19gd/35456.html  
4)Dongguan(东莞): http://www.tjcn.org/tjgb/19gd/35455.html  
5)Chongqing(重庆): http://www.tjcn.org/tjgb/22cq/35464.html  
The ***daily COVID-19 cases*** data is from https://github.com/canghailan/Wuhan-2019-nCoV and we did some adjustments by date.  
The ***outflow(from Wuhan)*** data is provided by China Unicom (cumulative value from January 1 to January 24, 2020).  
The ***search index*** data is collected from Baidu Index( http://index.baidu.com/v2/index.html#/).

### nature-data/coastal_cities.csv
It lists 57 coastal cities in China (Chinese name)

### nature-data/province_fix.csv 
It's the dummies data of 31 province (Except Hong Kong, Macau and Taiwan)

### nature-output/
This directory saves the project results and figures. 

### nature_model_submit.py
This Python script is to solve the problem of nonlinear model estimation. In this Paper, we proposed 2 types of **static models**(Exponential & Power static Model), 6 types of **dynamic models**(Exponential-Logistic model, Exponential-gompertz model, Exponential-Richards model, Power-Logistic model, Power-gompertz model, Power-Richards model), and 4 types of **dynamic increased models**(dF(X|t)/dt), and we used the LMFIT module to estimate the unknown parameters.

### nature_plot_submit.py
This Python script is to generate figures for the paper. Such as the 3D performance of the relationship among the confirmed cases, outflow from Wuhan and days to Jan. 23, the 2D performance of daily cumulative confirmed cases and outflow from Wuhan, the predicted and actual 
values of confirmed cases for each city, and the predicted and actual values of daily increased confirmed cases for each city.

