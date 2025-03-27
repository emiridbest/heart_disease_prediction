# https://www.kaggle.com/datasets/fajobgiua/enhanced-heart-disease-prediction-dataset/data
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
spark = SparkSession.builder.appName("HeartDisease").getOrCreate()
data = spark.read.csv('Heart_Disease.csv', header=True, inferSchema=True)

# Age Categorization
df = data.withColumn("AgeCategory",
    when(col("Age") < 40, 0)
    .when((col("Age") >= 40) & (col("Age") < 50), 1)
    .when((col("Age") >= 50) & (col("Age") < 60), 2)
    .when((col("Age") >= 60) & (col("Age") < 70), 3)
    .otherwise(4)
)

# Categorical columns for encoding


# Continuous columns for scaling
continuous_cols = ["RestingBP", "Cholesterol", "MaxHeartRate", "BMI"]
assembler = VectorAssembler(
    inputCols=continuous_cols,
    outputCol="continuous_features")
output = assembler.transform(df)

# Scale continuous_fetaure
scaler_data = StandardScaler(inputCol="continuous_features",
                        outputCol="scaled")
scaler = scaler_data.fit(output)
scaled_data = scaler.transform(output)

# Create indexers for categorical columns
# Stress Level Encoding
stress_indexer = StringIndexer(inputCol="StressLevel", outputCol="StressLevelIndex", handleInvalid="keep")
indexed = stress_indexer.fit(scaled_data).transform(scaled_data)

stress_encoder = OneHotEncoder(inputCol="StressLevelIndex", outputCol="StressLevelVec", dropLast=True)
encoded = stress_encoder.fit(indexed).transform(indexed)

categorical_cols = [
    "Sex", "ChestPainType", "FastingBloodSugar", 
    "ExerciseInducedAngina", "SmokingStatus", 
    "Diabetes", "PhysicalActivity", 
    "AgeCategory", "StressLevelVec"
]


categorical_assembler = VectorAssembler(
    inputCols=categorical_cols,
    outputCol="categorical_features"
)
assembled_cat = categorical_assembler.transform(encoded)
assembler = VectorAssembler(
    inputCols=["categorical_features","scaled"],
    outputCol="features")
final_data = assembler.transform(assembled_cat)
final_data = final_data.select("features", "HeartDisease")

train_data,test_data = final_data.randomSplit([0.7,0.3])


from pyspark.ml.classification import (
    LogisticRegression, 
    RandomForestClassifier, 
    GBTClassifier, 
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def evaluate_model(predictions):
    # Binary Classification Evaluator
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="HeartDisease", 
        metricName="areaUnderROC"
    )

    # Multiclass Classification Evaluator
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="HeartDisease", 
        predictionCol="prediction"
    )

    # Compute metrics
    metrics = {
        "AUC": binary_evaluator.evaluate(predictions),
        "Accuracy": multi_evaluator.evaluate(
            predictions, 
            {multi_evaluator.metricName: "accuracy"}
        ),
        "Precision": multi_evaluator.evaluate(
            predictions, 
            {multi_evaluator.metricName: "weightedPrecision"}
        ),
        "Recall": multi_evaluator.evaluate(
            predictions, 
            {multi_evaluator.metricName: "weightedRecall"}
        ),
        "F1 Score": multi_evaluator.evaluate(
            predictions, 
            {multi_evaluator.metricName: "weightedFMeasure"}
        )
    }
    
    return metrics

def plot_confusion_matrix(predictions):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Confusion Matrix (manual calculation)
    confusion_matrix = (
        predictions
        .groupBy("HeartDisease", "prediction")
        .count()
        .orderBy("HeartDisease", "prediction")
        .collect()
    )

    # Convert confusion_matrix to a DataFrame
    confusion_df = pd.DataFrame(confusion_matrix, columns=['Actual', 'Predicted', 'Count'])

    # Pivot the DataFrame to create a matrix
    confusion_matrix_array = confusion_df.pivot(
        index='Actual', 
        columns='Predicted', 
        values='Count'
    ).fillna(0).values

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_array, 
        annot=True, 
        cmap="Blues", 
        fmt='g',
        xticklabels=sorted(confusion_df['Predicted'].unique()),
        yticklabels=sorted(confusion_df['Actual'].unique())
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    return confusion_matrix

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features", 
        labelCol="HeartDisease"
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features", 
        labelCol="HeartDisease"
    ),
    "Gradient Boosted Trees": GBTClassifier(
        featuresCol="features", 
        labelCol="HeartDisease"
    ),
    "Decision Tree": DecisionTreeClassifier(
        featuresCol="features", 
        labelCol="HeartDisease"
    )
}

# Comparison dictionary
model_comparison = {}

# Train and evaluate each model
for name, classifier in classifiers.items():
    print(f"\nTraining {name}")
    
    # Fit the model
    model = classifier.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate metrics
    metrics = evaluate_model(predictions)
    
    # Plot confusion matrix
    confusion_matrix = plot_confusion_matrix(predictions)
    
    # Store results
    model_comparison[name] = {
        "Metrics": metrics,
        "Confusion Matrix": confusion_matrix
    }

# Print comparison
for model_name, results in model_comparison.items():
    print(f"\n{model_name} Performance:")
    for metric, value in results["Metrics"].items():
        print(f"{metric}: {value}")