## Predict-NYC-Taxi-Demand
This study explores New York City for-hire vehicle trip data using Apache Spark and machine learning techniques deployed on the SDSC Expanse platform. The dataset offers a large-scale view of travel behavior throughout the city, making it a valuable resource for examining transportation activity, traffic patterns, and changes in rider demand over time.

By analyzing trip characteristics and travel trends, the project aims to uncover insights that can improve transportation operations. Predictive models built from historical trip data can assist fleet operators in estimating travel times, identifying periods of increased demand, and improving resource utilization. Such analyses also contribute to broader efforts in urban transportation management and evidence-based decision making.

The scale of the data presents significant computational challenges. The 2017 FHV dataset contains more than 317 million records, resulting in data volumes that exceed the practical limits of conventional single-machine processing. Tasks such as transforming temporal variables, generating analytical features, performing large-scale aggregations, and training machine learning models require substantial computing resources. To address these challenges, Apache Spark was used to distribute computation across multiple nodes on SDSC Expanse, enabling efficient processing of the dataset and reducing execution time. This distributed approach provided the scalability necessary to analyze the data reliably and build predictive models on a dataset of this magnitude.

## Dataset

The analysis uses the 2017 NYC For-Hire Vehicle (FHV) trip dataset.

https://www.kaggle.com/datasets/verifiedbysuman/nyc-taxi & https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The dataset contains temporal, categorical, and location-based attributes that can be used to model trip behavior and duration.

##Data Exploration

### Main columns:
- dispatching_base_num
- pickup_datetime
- dropOff_datetime
- PUlocationID
- DOlocationID
- SR_Flag
- Affiliated_base_number

### Main exploration results:

- Total observations: 317,546,944
- dispatching_base_num unique values: 867
- Missing PUlocationID: 18,393,090
- Missing DOlocationID: 71,010,112
- Missing SR_Flag: 71,544,632
- Missing Affiliated_base_number: 3,760
- Distinct rows: 67,597,205
- Duplicate rows: 249,949,739
