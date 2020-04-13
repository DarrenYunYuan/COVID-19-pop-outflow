# COVID-2019-pop-outflow
## This is the Python project for Nature manuscript 2020-02-01778B  
Python development environment used: **Python 3.7.3**  
**numpy** version required: 1.16.2  
**pandas** version required: 0.24.2  
**matplotlib** version required: 3.2.0  
**lmfit** version required: 1.0.0

## Files description
### nature-data/pneumonia_panel_296_cities(submit).csv 
It's the basic data panel for our research, including 296 prefecture-level cities in China(based on CHINA CITY STATISTICAL YEARBOOK(2018)).  
The ***GDP and population*** data is collected from YEARBOOK, however, there are some missing values of GDP in the YEARBOOK, so we queried the GDP values of these cities in 2017 from the Chinese statistical information website(www.tjcn.org), details as following:  
1)Jiayuguan: http://www.tjcn.org/tjgb/28gs/35595.html  
2)Danzhou: http://www.tjcn.org/tjgb/21hn/35463.html  
3)Zhongshan: http://www.tjcn.org/tjgb/19gd/35456.html  
4)Dongguan: http://www.tjcn.org/tjgb/19gd/35455.html  
5)Chongqing: http://www.tjcn.org/tjgb/22cq/35464.html  
The ***daily COVID-19 cases*** data is from https://github.com/canghailan/Wuhan-2019-nCoV and we did some adjustments by date.  
The ***outflow(from Wuhan)*** data is provided by China Unicom (cumulative value from January 1 to January 24, 2020)
The ***search index*** data is collected from Baidu Index( http://index.baidu.com/v2/index.html#/).

### nature-data/coastal_cities.csv
It lists 57 coastal cities in China (Chinese name)

### nature-data/province_fix.csv 
It's the dummies data of 31 province (Except Hong Kong, Macau and Taiwan)

### nature-output
This directory saves the project results and figures.  


