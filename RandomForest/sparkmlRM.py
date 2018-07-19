from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql import  Row

spark=SparkSession.builder.appName("sparkmlRM").getOrCreate()


creditDF=spark.read.text("/Users/kaicat/server/spark-randomforest/data/germancredit.csv")
credit=creditDF.selectExpr("split(value,',') as col").selectExpr("cast (col[0] as double) as creditability","cast (col[1] as double) as balance", "cast( col[2] as double) as duration", "cast(col[3] as double) as history", "cast(col[4] as double) as purpose", "cast(col[5] as double) as amount", "cast (col[6] as double ) as  savings", "cast(col[7] as double) as employment", "cast(col[8] as double) as instPercent", "cast(col[9] as double ) as sexMarried","cast(col[10] as double) as  guarantors", "cast(col[11] as double) as residenceDuration", "cast(col[12] as double ) as assets", "cast(col[13] as double) as age", "cast(col[14] as double ) as concCredit", "cast( col[15] as double ) as apartment", "cast(col[16] as double ) as credits", "cast(col[17] as double ) as occupation", "cast(col[18] as double ) as dependents", "cast(col[19] as double ) as hasPhone", "cast(col[20] as double ) as foreign")

credit.printSchema()

featureCols =["balance", "duration", "history", "purpose", "amount", "savings", "employment", "instPercent", "sexMarried",  "guarantors", "residenceDuration", "assets",  "age", "concCredit", "apartment", "credits",  "occupation", "dependents",  "hasPhone", "foreign"]

lindexer=StringIndexer().setInputCol("creditability").setOutputCol("label")

assembler=VectorAssembler().setInputCols(featureCols).setOutputCol("features")

pipeline=Pipeline().setStages([assembler,lindexer])

credit=pipeline.fit(credit).transform(credit)

(training,test)=credit.randomSplit([0.7,0.3],seed=1234)

classifier=RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(1234)
model=classifier.fit(training)

predictions=model.transform(test)

predictions.show()

evaluator=BinaryClassificationEvaluator().setLabelCol("label")

accuracy=evaluator.evaluate(predictions)


print("accuracy= %f" % accuracy)
