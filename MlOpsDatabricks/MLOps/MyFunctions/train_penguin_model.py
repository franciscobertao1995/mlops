def train_penguin_model(training_data, test_data, maxIterations, regularization):
    import mlflow
    import mlflow.spark
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import time
   
    # Start an MLflow run
    with mlflow.start_run():
   
        catFeature = "Island"
        numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
        # Define the feature engineering and model steps
        catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
        numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
        numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
        featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
        algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=maxIterations, regParam=regularization)
   
        # Chain the steps as stages in a pipeline
        pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
        # Log training parameter values
        print ("Training Logistic Regression model...")
        mlflow.log_param('maxIter', algo.getMaxIter())
        mlflow.log_param('regParam', algo.getRegParam())
        model = pipeline.fit(training_data)
   
        # Evaluate the model and log metrics
        prediction = model.transform(test_data)
        metrics = ["accuracy", "weightedRecall", "weightedPrecision"]
        for metric in metrics:
            evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName=metric)
            metricValue = evaluator.evaluate(prediction)
            print("%s: %s" % (metric, metricValue))
            mlflow.log_metric(metric, metricValue)
   
   
        # Log the model itself
        unique_model_name = "classifier-" + str(time.time())
        mlflow.spark.log_model(model, unique_model_name, mlflow.spark.get_default_conda_env())
        modelpath = "/model/%s" % (unique_model_name)
        mlflow.spark.save_model(model, modelpath)
   
        print("Experiment run complete.")