########## Setup Cluster ######################################################################################################
#databricks clusters create --cluster-name my-cluster --spark-version 9.0.x-scala2.12 --node-type Standard_DS3_v2 --num-workers 2


########## Connect to Databricks ######################################################################################################
import databricks.koalas as ks
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Databricks Example") \
    .getOrCreate()

# Read data into a Koalas DataFrame
df = ks.read_csv("dbfs:/FileStore/shared_uploads/sample_data.csv")



########## Manipulate with Koalas ######################################################################################################
# Show first few rows of the DataFrame
df.head()

# Filter data
filtered_df = df[df['column'] > 10]

# Group by and aggregate
grouped_df = df.groupby('column').agg({'other_column': 'mean'})

# Join with another DataFrame
joined_df = df.join(other_df, on='key_column', how='inner')



########## Visualize ######################################################################################################
import matplotlib.pyplot as plt

# Plot histogram
plt.hist(df['column'], bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Column')
plt.show()



########## Make Predictions ######################################################################################################
from pyspark.ml.regression import LinearRegression

# Train linear regression model
lr = LinearRegression(featuresCol='features', labelCol='label')
model = lr.fit(df)

# Make predictions
predictions = model.transform(df)


########## Schedule ######################################################################################################
#databricks jobs create --name my-job --notebook my-notebook --schedule "0 0 * * *" --max-retries 3
