### Students: 
[Andrey Dodon](https://www.kaggle.com/andreydodon)</br>
[Doron Tzur](https://www.kaggle.com/andreydodon) 

# Introduction

In the course of our postgraduate pursuit, we undertook a project labelled "Evaluating the COVID-19 
Influence on Air Quality in Israel: A Machine Learning Perspective." The inspiration for this project 
was derived from a previously published investigation that delved into a related subject - 
[Assessing the COVID-19 Impact on Air Quality: A Machine Learning Approach](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091202). 
Numerous research undertakings globally have probed into the consequences of the lockdown measures enforced 
in 2020 due to the outbreak of the Corona Virus disease on air quality. Although a common consensus is 
reached on the diminution of pollution, there persists a dispute over the most dependable methodology 
for gauging the reduction in pollution.

In our research, we employed machine learning models built on the Gradient Boosting Machine algorithm 
to measure the impact of the pandemic on the quality of air in various districts of Israel. We initially 
validated the precision of our forecasts by performing cross-validation over the five-year span preceding 
the lockdown.

Following this, we quantified the shifts in pollution levels during the period of the lockdown. 
Our scrutiny divulged that districts with high vehicular congestion registered the most dramatic decrease 
in pollution. As the restrictions were partially revoked, the pollution concentrations started reverting 
to pre-pandemic figures. We buttressed our estimation of the fall in pollution by evaluating the confidence 
of our predictions.

Our objective through this project was to augment the understanding of how the COVID-19 pandemic 
has influenced the quality of air in Israel. By leveraging machine learning techniques, we offered 
valuable insights into the patterns and shifts of pollution during the lockdown phase. Our deductions 
underscore the significance of executing efficient strategies to curb pollution, especially in regions 
with heavy traffic. Moreover, our research accentuates the potential of machine learning algorithms in 
analysing and forecasting the dynamics of air quality amid unprecedented occurrences such as a pandemic.

# Methodology
Our research methodology employed machine learning models, specifically the Gradient Boosting Machine (GBM) 
algorithm, to examine the pandemic's impact on air quality across various districts in Israel. The model 
was initially tested for accuracy through cross-validation across a three-year span before the lockdown.

Here is a detailed breakdown of the machine learning model parameters and methods:

* The model was built using a Gradient Boosting Classifier.
* The model construction involved 100 decision trees (parameter n_estimators=100), a learning rate of 0.1 
(parameter learning_rate=0.1), and a maximum tree depth of 2 (parameter max_depth=2).
* Each decision tree was trained on a random sample consisting of 80% of the data (parameter subsample=0.8). 
This subsampling approach adds an element of randomness to model training, aiding in the prevention of 
overfitting.
* We employed 5-fold cross-validation to assess the model's performance. This technique divides the dataset 
into five subsets, using four for training and the remaining one for validation. The process is repeated 
five times, ensuring each subset is used once for validation. Cross-validation offers a more reliable measure 
of the model's performance and its ability to generalize to unseen data.

Following the model's validation, we quantified the changes in pollution levels during the lockdown period. 
Our analysis revealed that districts with high vehicular congestion experienced the most significant reduction 
in pollution. As lockdown restrictions were gradually lifted, pollution levels began to approach pre-pandemic 
figures. We reinforced our estimates of the pollution decrease by evaluating the confidence of our predictions.

The primary objective of this project was to enhance our understanding of the COVID-19 pandemic's impact on air 
quality in Israel. Leveraging machine learning techniques allowed us to gain valuable insights into pollution 
patterns and shifts during the lockdown. Our findings emphasize the importance of implementing effective 
strategies to mitigate pollution, particularly in heavily trafficked areas. Furthermore, our research highlights 
the potential of machine learning algorithms in analyzing and predicting air quality dynamics during 
extraordinary events such as a pandemic.

# Dataset

The dataset utilized in our study amalgamates data from several sources, primarily from the Ministry of 
Environmental Protection of Israel and the Meteorological Service of Israel. Unfortunately, we encountered 
discrepancies across different data fields, prompting us to focus on these two specific sources. The Israel 
Central Bureau of Statistics' databank, while considered, was excluded due to the substantial costs associated 
with data cleaning and engineering.

From the Ministry of Environmental Protection, we procured essential pollution measurements such as Nitric Oxide (NO), 
Nitrogen Oxides (NOX), Nitrogen Dioxide (NO2), and Sulfur Dioxide (SO2). The Meteorological Service of Israel provided 
us with data for Wind Speed (WS), Wind Direction (WD), and Relative Humidity (RH), features that have been strongly 
endorsed in previous studies.

Additionally, we included categorical features such as the day of the week, area, and lockdown status into our dataset. 
These categorical variables offer additional contextual information that can help improve the accuracy of our model's 
predictions.

To accurately measure pollution levels during both lockdown and non-lockdown periods, we utilized the aforementioned 
features which are all publicly available, thereby allowing for the transparency and reproducibility of our study. 
This public availability ensures that our research can be scrutinized, replicated, and further built upon by other 
researchers, thereby contributing to a broader and more collaborative scientific understanding of pollution dynamics 
in relation to human activities.

![](img/diagram.png)


# Features:
* **NO:** Stands for Nitric Oxide, a colorless and poisonous gas that is a primary contributor to 
atmospheric pollution. It's primarily produced from the emissions of vehicles and power plants.

* **NOX:** This is a generic term for the nitrogen oxides that are most relevant for air pollution, 
namely nitric oxide (NO) and nitrogen dioxide (NO2). These gases contribute to the formation of smog and 
acid rain, as well as affecting tropospheric ozone.

* **NO2:** Nitrogen Dioxide is a reddish-brown gas with a characteristic sharp, biting odor and is a 
prominent air pollutant. It is produced when nitric oxide (NO) combines with oxygen (O2).

* **SO2:** Sulfur Dioxide is a colorless gas with a strong, choking odor. It is released from the burning 
of fossil fuels (coal and oil) and the smelting of mineral ores that contain sulfur. High levels of sulfur 
dioxide can cause inflammation and irritation of the respiratory system, especially during heavy physical 
activity.

* **WS:** Wind Speed is a measurement of how fast the air is moving in a certain area. It's an important 
factor in weather forecasts and climatic studies. It can also affect the dispersion of pollutants in the 
atmosphere.

* **WD:** Wind Direction is the direction from which the wind is coming. It's usually reported in cardinal 
directions or in degrees. For example, a wind coming from the north is a north wind, and a wind coming from 
the west is a west wind.

* **RH:** Relative Humidity is the amount of moisture in the air compared to the maximum amount of moisture 
the air could hold at the same temperature. It's expressed as a percentage. High relative humidity can enhance 
the formation and persistence of certain air pollutants.

* **Lockdown:** This is a binary feature representing whether a COVID-19 lockdown was in effect on a 
particular day. A value of 1 indicates that a lockdown was in effect, while a 0 indicates no lockdown. 
This feature is crucial in this analysis as it allows us to distinguish between periods of normal activity 
and those with reduced human activity due to lockdowns. It helps us understand the impact of lockdowns on the 
levels of air pollutants and overall air quality.

* **Area:** This categorical feature represents the geographic region where the air quality measurements 
were taken. Different areas can have varying levels of pollutants due to factors such as population density, 
industrial activity, vehicular traffic, among others. 

* **Day of Week:** This is a categorical feature that indicates the day of the week when the measurements 
were taken. It is represented as an integer, where 0 stands for Monday and 6 stands for Sunday. Including 
this feature allows us to capture any weekly patterns in air quality. 

# Explanatory
Each of the features mentioned - **NO**, **NOX**, **NO2**, **SO2**, **WS**, **WD**, and **RH** - can provide 
valuable insights in the context of an air quality analysis. Here's why:

* **NO, NOX, NO2, SO2:** These are key pollutants that directly impact air quality. During COVID-19 lockdowns, 
many anthropogenic activities, such as driving vehicles or operating industrial plants, were significantly reduced. 
This potentially leads to a reduction in the emission of these pollutants. By tracking the levels of these 
pollutants, we can quantify the impact of lockdown measures on air quality. Machine learning can help identify 
patterns and correlations, predict future pollution levels, and even identify the sources of these pollutants.

* **WS (Wind Speed):** Wind can disperse or accumulate air pollutants, affecting air quality locally and 
regionally. During lockdowns, any changes in air pollution levels might be influenced not only by reduced human 
activities but also by weather conditions such as wind speed. ML can help in understanding the role of wind 
speed in the dispersion of pollutants and predicting how changes in wind patterns might affect future air 
quality.

* **WD (Wind Direction):** Similar to wind speed, wind direction can significantly influence where pollutants 
are carried, potentially impacting air quality in different areas. Machine learning models can help analyze 
the impact of wind direction on pollutant distribution during the lockdown period.

* **RH (Relative Humidity):** Moisture in the air can react with certain pollutants, potentially leading to 
secondary pollutants or exacerbating pollution levels. High humidity can also trap pollutants close to the 
ground. Incorporating RH into ML models allows for a more comprehensive understanding of air quality dynamics 
during the lockdown.

* **Lockdown:** This is a binary variable indicating whether a lockdown was in effect on a given day. The 
lockdown measures imposed due to COVID-19 had a profound effect on human behavior, leading to a dramatic 
reduction in anthropogenic activities such as vehicular movement, industrial operations, and other economic 
activities. This abrupt halt or scale-down of activities resulted in a significant decrease in emissions of 
various pollutants. Thus, the lockdown variable serves as an essential indicator of the periods during which 
these extraordinary conditions were in effect. By incorporating this variable into our model, we can directly 
evaluate the changes in air quality attributable to the lockdown periods versus the non-lockdown periods.

* **Area:** This categorical variable represents different geographic regions where the measurements were taken. 
Different regions might have varying levels of pollutants due to differences in population density, industrial 
activities, vehicular movement, and other factors. By including the 'area' feature in our model, we can capture 
these spatial differences in air quality and investigate whether the impact of the lockdown varied across 
different regions.

* **Day of Week:** This categorical variable represents the day of the week for each observation. 
Human activities and thus air pollution levels can exhibit weekly cycles. For instance, traffic volume and 
industrial activities might be lower on weekends compared to weekdays, leading to lower emissions. 
By including this feature, our model can capture such weekly patterns in air pollution.

# Results
### **Lockdown model performance**

| Fold | Accuracy    | Precision   | Recall      | F1 Score    | AUC         |
|------|-------------|-------------|-------------|-------------|-------------|
| 1    | 0.921892189 | 0.895287958 | 0.91689008  | 0.905960265 | 0.978519767 |
| 2    | 0.913091309 | 0.8828125   | 0.908847185 | 0.895640687 | 0.9722225   |
| 3    | 0.927392739 | 0.898701299 | 0.927613941 | 0.91292876  | 0.981553359 |
| 4    | 0.941694169 | 0.914507772 | 0.946380697 | 0.930171278 | 0.986007463 |
| 5    | 0.921806167 | 0.888888889 | 0.924731183 | 0.906455863 | 0.976481002 |

### Non-lockdown model performance

| Fold | Accuracy    | Precision   | Recall      | F1 Score    | AUC         |
|------|-------------|-------------|-------------|-------------|-------------|
| 1    | 0.939050153 | 0.941194821 | 0.937538844 | 0.939363275 | 0.985989308 |
| 2    | 0.93404272  | 0.941436464 | 0.926662523 | 0.933991074 | 0.984944212 |
| 3    | 0.938262911 | 0.938899254 | 0.938461538 | 0.938680345 | 0.985791564 |
| 4    | 0.936306729 | 0.940724479 | 0.932245532 | 0.936465813 | 0.98641859  |
| 5    | 0.93685446  | 0.943717484 | 0.930080796 | 0.936849519 | 0.985678944 |

The ensuing results represent performance metrics for each of the five cross-validation folds implemented during 
both the lockdown and non-lockdown periods. Each column in the dataset provides specific insight into the model's 
effectiveness:

The **Fold** column indicates the specific number of the fold within the cross-validation process. 
Cross-validation, a universally recognized technique, measures a model's capability to generalize to 
independent data sets. This technique entails dividing the dataset into 'folds' (five folds in this case) 
and repeatedly training and testing the model, with each iteration using a different fold as the test set.

The **Accuracy** metric denotes the ratio of correctly predicted instances (comprising both true positives 
and true negatives) to the total instances evaluated. This provides a measure of the overall correctness of 
the model's predictions.

**Precision** corresponds to the proportion of true positives over the sum of true positives and false positives. 
This metric assesses the accuracy of the model's positive predictions.

**Recall**, also known as sensitivity, is the fraction of true positives to the sum of true positives and false 
negatives. This metric gauges the model's ability to correctly identify all positive instances.

The **F1 Score** is the harmonic mean of precision and recall, aiming to balance these two metrics and providing a 
composite measure of model quality.

**AUC**, or Area Under the ROC Curve, is a comprehensive performance indicator. The ROC curve graphically 
presents the true positive rate against the false positive rate at various potential thresholds of a binary classifier. 
The AUC calculates the two-dimensional area under the entire ROC curve, providing an aggregate performance measure 
across all potential classification thresholds. An AUC value of 1 indicates perfect predictions, whereas a value of 
0 implies total misclassification.

In the supplied data, each row signifies the performance of the model for each fold during both the lockdown 
and non-lockdown periods. The model exhibits consistent performance across all folds, as evidenced by the accuracy, 
F1 score, and AUC values, all of which exceed 90%.

# Jewish Holidays Hypothesis
During the course of our study, we observed an intriguing phenomenon: a noticeable decrease in pollutant levels 
coinciding with certain Jewish holidays. This effect appears to mirror, to a certain extent, the pollution reductions 
witnessed during the COVID-19 lockdowns.

Our initial hypothesis was that these reductions in pollutant levels were a result of decreased human activities during 
Jewish holidays, an effect similar to that observed during lockdowns. This notion is not far-fetched, considering that 
many businesses shut down and a substantial segment of the population abstains from driving or operating machinery 
during Jewish holidays in observance of religious traditions.

However, we have now refined this hypothesis to propose that specific activities, usually paused during Jewish holidays 
but continuing through the COVID-19 lockdowns, may significantly contribute to these observed pollutant levels. This 
implies that Jewish holidays could result in a more pronounced decrease in pollution compared to the lockdowns due to 
the more comprehensive suspension of certain pollution-generating activities.

To investigate this hypothesis further, we intend to extend our research model to include Jewish holidays as an 
additional categorical feature. This will enable us to evaluate the impact of these holidays on air quality and 
contrast it with the effects of the COVID-19 lockdowns. Moreover, it becomes essential to identify these unique 
activities halted during Jewish holidays and quantify their typical contribution to pollution levels.

The Jewish holidays we aim to consider encompass Rosh Hashanah (Jewish New Year), Yom Kippur (Day of Atonement), 
Sukkot (Feast of Tabernacles), Hanukkah (Festival of Lights), and Passover. The relevance of these holidays lies in 
their stringent observances that significantly alter daily routines, including restrictions on work, travel, and the 
use of electrical appliances, among others.

The integration of Jewish holidays into our model will assist in discerning whether and to what extent temporary 
cessations of human activities during these periods affect air quality. By juxtaposing the effects of these holidays 
with the impact of the COVID-19 lockdowns, we aim to deepen our understanding of the relationship between human 
activities and air pollution, specifically in Israel.

![](img/NO_levels_specific_periods.png)
![](img/NO_levels_stem_plot.png)

If validated, this hypothesis could have considerable implications for air quality management strategies. Policies 
encouraging the voluntary reduction of specific activities during designated periods could be implemented to improve 
air quality. This approach would not only provide a broader perspective on how behavioral changes influenced by 
cultural or religious practices can impact the environment but also offer valuable insights for policymakers, 
environmentalists, and the public.

We posit that the exploration of this hypothesis will contribute to the scientific understanding of factors affecting 
air quality and propose strategies that lead to cleaner air and a healthier environment. However, the sensitivity of 
religious observances must be recognized, and communities must be actively involved in discussions about potential 
modifications to routines. Our ultimate goal is to promote strategies that strike a balance between cultural respect 
and environmental health.


# Planned Enhancements and Future Directions
As we continue to refine our model, we plan on incorporating additional data sets and applying advanced machine 
learning techniques to further improve its performance and predictive capabilities. A specific area of interest 
includes the exploration of H2O AutoML, a versatile automated machine learning platform. The potential integration 
of this tool could significantly enhance our model's efficacy by automating the machine learning workflow, enabling 
us to analyze more complex patterns and relationships within our data.

Additionally, we aim to investigate the impact of post-lockdown activities on air quality, recognizing that the 
resumption of typical routines could potentially erode the improved air quality levels observed during lockdown periods. 
A granular understanding of these impacts could help inform policymaking efforts, thereby maintaining these enhanced 
air quality levels and positively affecting the health of the population.

One area we plan to delve into with greater depth is the effect of Jewish holidays on air pollution levels. By 
including these events as significant categorical variables in our model, we aim to understand the extent to which 
these religious observances, and the subsequent reduction in specific activities, contribute to improved air quality. 
These inquiries would extend beyond traditional "business as usual" scenarios to uncover how religious and cultural 
practices can positively impact the environment. We hope to leverage these insights to recommend policies that respect 
these traditions while also benefiting the environment.

The work conducted and presented thus far establishes a promising trajectory for future research into the environmental 
repercussions of extraordinary events, including pandemics, religious holidays, and economic downturns. Understanding 
these impacts is critical in paving the way towards mitigating adverse effects, implementing effective pollution 
reduction strategies, and ultimately enhancing the quality of life for all residents.

In sum, we believe these planned enhancements and directions for future research will significantly contribute to the 
evolving field of environmental management, offering valuable insights for policymakers, environmentalists, and the 
public at large.

## Inclusion of Highway Proximity in Air Quality Analysis
As part of our dataset augmentation, we are considering the introduction of a new feature: the distance from an air 
quality monitoring station to the nearest highway. This feature presents multiple benefits:

**Emissions Source Proximity:** Highways, being major vectors for vehicular emissions, significantly contribute to 
the concentration of air pollutants such as NO, NO2, NOX, and SO2. Stations in proximity to highways are expected 
to register heightened pollutant concentrations.

**Localized Pollution Impact:** Pollution exhibits a localized nature. The influence of vehicular emissions might 
not uniformly disperse across extensive areas. Factoring in a station's highway proximity facilitates a more accurate 
depiction of localized pollution scenarios.

**Lockdown Impact Assessment:** The reduction in highway traffic as a result of COVID-19 lockdowns could have affected 
air quality more profoundly near highways. Integrating a highway proximity feature could assist in quantifying the 
effects of lockdown initiatives on air quality, particularly for stations near major roads.

**Model Granularity Enhancement:** Incorporating the highway proximity feature could lend an additional layer of 
granularity to our model, enabling it to capture more nuanced spatial patterns in pollution levels.

Nonetheless, the incorporation of this feature is accompanied by certain challenges:

**Data Accuracy:** For this feature to be effective, precise and reliable data on highway locations and their 
corresponding distances from each station are indispensable.

**Interplay of Multiple Factors:** Highway proximity is only one of the myriad factors influencing pollution levels. 
Other factors, including wind direction, topography, and local emission sources, also significantly contribute and 
must be taken into consideration.

**Model Complexity Trade-Off:** Introducing this feature could increase model complexity, potentially complicating 
interpretation. It's crucial to maintain a balance between the potential for enhanced prediction accuracy and the 
risk of increased model complexity.
