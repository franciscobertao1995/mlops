# Download CSV file to DBFS
dbutils.fs.rm("dbfs:/mlflow_lab", recurse=True)
dbutils.fs.mkdirs("dbfs:/mlflow_lab")
dbutils.fs.cp("https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv", "dbfs:/mlflow_lab/penguins.csv")

# Create dataframe and split the train and test data

from pyspark.sql.types import *
from pyspark.sql.functions import *

from MyFunctions import train_penguin_model
   
data = spark.read.format("csv").option("header", "true").load("/mlflow_lab/penguins.csv")
data = data.dropna().select(col("Island").astype("string"),
                            col("CulmenLength").astype("float"),
                            col("CulmenDepth").astype("float"),
                            col("FlipperLength").astype("float"),
                            col("BodyMass").astype("float"),
                            col("Species").astype("int")
                          )
display(data.sample(0.2))
   
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
print ("Training Rows:", train.count(), " Testing Rows:", test.count())

train_penguin_model(train, test, 10, 0.2)
