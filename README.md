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

## Figures
- Figure 1. Top 10 Pickup Locations. Trip counts for the most frequent pickup locations in the Spark-aggregated dataset.
- Figure 2. Histogram of Pickup Location IDs. Distribution of pickup location IDs from a Spark sample.
- Figure 3. Histogram of Trip Duration. Distribution of trip duration after filtering invalid durations.
- Figure 4. Pickup Hour vs. Trip Duration. Sampled scatter plot showing temporal variation in trip duration.
- Figure 5. PCA Explained Variance. Individual and cumulative explained variance ratios for the retained principal components.
- Figure 6. Model Comparison. Accuracy, F1, and AUC across Random Forest, full Logistic Regression, and PCA + Logistic Regression.
- Figure 7. Prediction Analysis. Example correct classifications, false positives, and false negatives from the test set.

  ## Methods

  ### Data Preprocessing and Feature Engineering


```
df = spark.read.parquet(data_path)
print("Number of observations:", df.count())
df.printSchema()

categorical_cols = [c for c, dtype in df.dtypes if dtype in ["string", "boolean"]]
for c in categorical_cols:
    print(c, df.select(c).distinct().count())
Preprocessing (using Spark)
Preprocessing was completed in Spark only. Rows with missing pickup_datetime or dropOff_datetime were removed. A new feature, trip_duration_minutes, was created from the timestamp difference. Invalid durations were filtered out, and a binary label was created:

label = 1 if trip_duration_minutes >= 30
label = 0 otherwise
Additional time-based features were engineered:

pickup_hour
pickup_month
pickup_dayofweek
is_weekend
Missing location IDs were filled with -1.

df_clean = df.dropna(subset=["pickup_datetime", "dropOff_datetime"])
df_clean = df_clean.withColumn(
    "trip_duration_minutes",
    (unix_timestamp(col("dropOff_datetime")) - unix_timestamp(col("pickup_datetime"))) / 60.0
).filter(
    (col("trip_duration_minutes") > 0) &
    (col("trip_duration_minutes") <= 180)
).withColumn(
    "label",
    when(col("trip_duration_minutes") >= 30, 1).otherwise(0)
)
```

Key preprocessing steps include:

- Removal of records with missing timestamps
- Creation of trip duration in minutes
- Filtering invalid trip durations
- Construction of a binary classification target
- Handling missing location identifiers
- Feature engineering from timestamp information

Generated temporal features include:

- Pickup hour
- Pickup month
- Day of week
- Weekend indicator

### Final modeling features:

PUlocationID
DOlocationID
pickup_hour
pickup_month
pickup_dayofweek
is_weekend

### Distributed Model Training

Model 1: First Distributed Model
The first distributed model was a Spark RandomForestClassifier. Two configurations were tested:

RF1: numTrees=20, maxDepth=5, maxBins=300
RF2: numTrees=50, maxDepth=10, maxBins=300


```rf2 = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    maxDepth=10,
    maxBins=300,
    seed=42
)
```
Pipeline 2-PCA + Logistic Regression
Compares a full-feature baseline Logistic Regression model against an unsupervised PCA system that compresses features into the 3 most significant latent components before classification.
- Standardization: Features were normalized using StandardScaler.
- Baeline model-A baseline Logistic Regression model was trained on the full standardized feature set
- Dimensionality Reduction: Principal Component Analysis (PCA) condensed the feature space to the 3 most significant components, retaining 99.9% of cumulative data variance.
- Classification: A distributed L2-regularized LogisticRegression model was trained on the PCA components.
  
 The processed Spark sample had 1,193,817 rows with split sizes:
 - Train: 836,353
 - Validation: 178,705
 - Test: 178,759


For the final PCA workflow, the sampled class counts were:

label = 0: 240,122
label = 1: 58,804
Split sizes:
Train: 209,266
Validation: 44,659
Test: 45,001


```

k = 3
pca = PCA(k=3, inputCol="features_scaled", outputCol="pca_features")

lr_pca = LogisticRegression(
    labelCol="label",
    featuresCol="pca_features",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0

```


