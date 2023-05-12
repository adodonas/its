### Students: 
[Andrey Dodon](https://www.kaggle.com/andreydodon)</br>
[Doron Tzur](https://www.kaggle.com/andreydodon) 

# Introduction

In the course of our postgraduate pursuit, we undertook a project labelled "Evaluating the COVID-19 Influence on Air Quality in Israel: A Machine Learning Perspective." The inspiration for this project was derived from a previously published investigation that delved into a related subject - [Assessing the COVID-19 Impact on Air Quality: A Machine Learning Approach](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091202). Numerous research undertakings globally have probed into the consequences of the lockdown measures enforced in 2020 due to the outbreak of the Corona Virus disease on air quality. Although a common consensus is reached on the diminution of pollution, there persists a dispute over the most dependable methodology for gauging the reduction in pollution.

In our research, we employed machine learning models built on the Gradient Boosting Machine algorithm to measure the impact of the pandemic on the quality of air in various districts of Israel. We initially validated the precision of our forecasts by performing cross-validation over the five-year span preceding the lockdown.

Following this, we quantified the shifts in pollution levels during the period of the lockdown. Our scrutiny divulged that districts with high vehicular congestion registered the most dramatic decrease in pollution. As the restrictions were partially revoked, the pollution concentrations started reverting to pre-pandemic figures. We buttressed our estimation of the fall in pollution by evaluating the confidence of our predictions.

Our objective through this project was to augment the understanding of how the COVID-19 pandemic has influenced the quality of air in Israel. By leveraging machine learning techniques, we offered valuable insights into the patterns and shifts of pollution during the lockdown phase. Our deductions underscore the significance of executing efficient strategies to curb pollution, especially in regions with heavy traffic. Moreover, our research accentuates the potential of machine learning algorithms in analysing and forecasting the dynamics of air quality amid unprecedented occurrences such as a pandemic.


# Dataset

Here's a clear and concise summary of the data we are working with. During our thorough examination of multiple data sources, we encountered inconsistencies across various data fields. Consequently, we opted to gather data from specific databanks, namely the Ministry of Environmental Protection of Israel and the Meteorological Service of Israel. We also explored the databank of the Israel Central Bureau of Statistics but deemed it unsuitable due to the substantial costs associated with data cleaning and engineering.

From the Ministry of Environmental Protection, we obtained principal pollution measures such as NO, NOX, NO2, SO2 (detailed explanations for each will follow). On the other hand, we sourced the WS, WD, and RH features from the Meteorological Service of Israel, features which have been highly recommended based on past studies. In addition to these, we incorporated categorical features such as the day of the week and area into our dataset.

![](img/diagram.png)


# Features:
* **NO:** Stands for Nitric Oxide, a colorless and poisonous gas that is a primary contributor to atmospheric pollution. It's primarily produced from the emissions of vehicles and power plants.

* **NOX:** This is a generic term for the nitrogen oxides that are most relevant for air pollution, namely nitric oxide (NO) and nitrogen dioxide (NO2). These gases contribute to the formation of smog and acid rain, as well as affecting tropospheric ozone.

* **NO2:** Nitrogen Dioxide is a reddish-brown gas with a characteristic sharp, biting odor and is a prominent air pollutant. It is produced when nitric oxide (NO) combines with oxygen (O2).

* **SO2:** Sulfur Dioxide is a colorless gas with a strong, choking odor. It is released from the burning of fossil fuels (coal and oil) and the smelting of mineral ores that contain sulfur. High levels of sulfur dioxide can cause inflammation and irritation of the respiratory system, especially during heavy physical activity.

* **WS:** Wind Speed is a measurement of how fast the air is moving in a certain area. It's an important factor in weather forecasts and climatic studies. It can also affect the dispersion of pollutants in the atmosphere.

* **WD:** Wind Direction is the direction from which the wind is coming. It's usually reported in cardinal directions or in degrees. For example, a wind coming from the north is a north wind, and a wind coming from the west is a west wind.

* **RH:** Relative Humidity is the amount of moisture in the air compared to the maximum amount of moisture the air could hold at the same temperature. It's expressed as a percentage. High relative humidity can enhance the formation and persistence of certain air pollutants.

## Explanatory

Each of the features mentioned - **NO**, **NOX**, **NO2**, **SO2**, **WS**, **WD**, and **RH** - can provide valuable insights in the context of an air quality analysis. Here's why:

* **NO, NOX, NO2, SO2:** These are key pollutants that directly impact air quality. During COVID-19 lockdowns, many anthropogenic activities, such as driving vehicles or operating industrial plants, were significantly reduced. This potentially leads to a reduction in the emission of these pollutants. By tracking the levels of these pollutants, we can quantify the impact of lockdown measures on air quality. Machine learning can help identify patterns and correlations, predict future pollution levels, and even identify the sources of these pollutants.

* **WS (Wind Speed):** Wind can disperse or accumulate air pollutants, affecting air quality locally and regionally. During lockdowns, any changes in air pollution levels might be influenced not only by reduced human activities but also by weather conditions such as wind speed. ML can help in understanding the role of wind speed in the dispersion of pollutants and predicting how changes in wind patterns might affect future air quality.

* **WD (Wind Direction):** Similar to wind speed, wind direction can significantly influence where pollutants are carried, potentially impacting air quality in different areas. Machine learning models can help analyze the impact of wind direction on pollutant distribution during the lockdown period.

* **RH (Relative Humidity):** Moisture in the air can react with certain pollutants, potentially leading to secondary pollutants or exacerbating pollution levels. High humidity can also trap pollutants close to the ground. Incorporating RH into ML models allows for a more comprehensive understanding of air quality dynamics during the lockdown.

Finally, the **day of the week** and the **area** are also useful features. Lockdown measures may vary by area and day of the week, which can influence human activities and thus air pollution levels. Machine learning can help identify these location-specific and time-specific patterns.

By leveraging these features, ML models can provide a more accurate and detailed picture of how COVID-19 lockdowns have impacted air quality.