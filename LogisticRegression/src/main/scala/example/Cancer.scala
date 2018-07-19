package lrexample1 


import org.apache.spark._

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.SparkSession

/*
 * https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
 * Array(1000025,5,1,1,1,2,1,3,1,1,2)
   0. Sample code number            id number 
   1. Clump Thickness               1 - 10
   2. Uniformity of Cell Size       1 - 10
   3. Uniformity of Cell Shape      1 - 10
   4. Marginal Adhesion             1 - 10
   5. Single Epithelial Cell Size   1 - 10
   6. Bare Nuclei                   1 - 10
   7. Bland Chromatin               1 - 10
   8. Normal Nucleoli               1 - 10
   9. Mitoses                       1 - 10
  10. Class:                        (2 for benign, 4 for malignant)
 */

object Cancer {

  case class cancerclass(id: Double, thickness: Double, size: Double, shape: Double, madh: Double, epsize: Double, bnuc: Double, bchrom: Double, nNuc: Double, mit: Double, clas: Double)

  def main(args: Array[String]) {

    // create a spark session
    val spark=SparkSession.builder().appName("sparkml-cancer").getOrCreate()
    import spark.implicits._ 

    // define the schema for cancer data
    val cancerschema=Encoders.product[cancerclass].schema

    // load the csv file to the dataframe cancerDF and drop the rows having null values
    val cancerDF=spark.read.format("csv").option("header","false").schema(cancerschema).load("/Users/kaicat/server/spark-ml-lr/data/wbcd.csv")
    val cancernaDF=cancerDF.na.drop()
    cancernaDF.createOrReplaceTempView("cancerview")
   
    // keep label and feature attributes only
    val cancerdataDF=spark.sql("select  clas,thickness,size,shape,madh,epsize,bnuc,bchrom,nNuc,mit from cancerview")

    //store cancerDF to cache
    cancerdataDF.cache()

    //extract features and assembler feature inputs into one vector

    val featureCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    
    // using StringIndexer to make clas input to 0 and 1
     val lindexer=new StringIndexer().setInputCol("clas").setOutputCol("label")
     

    val splitSeed = 1234
    val Array(trainData, testData) = cancerdataDF.randomSplit(Array(0.7, 0.3), splitSeed)

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    
    
    // initaliaze a pipeline
     val pipeline= new Pipeline().setStages(Array(assembler,lindexer,lr))   

    val model = pipeline.fit(trainData)
 
    val predictionsDF = model.transform(testData)
    predictionsDF.select("clas", "label", "prediction").show(5)

    //Evaluates predictions and returns a underareaROC(larger is better)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
   
    val accuracy = evaluator.evaluate(predictionsDF)


    val lp = predictionsDF.select( "label", "prediction")
    val counttotal = predictionsDF.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
    val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
    val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    println(s"counttotal: ${counttotal} correct: ${correct} wrong: ${wrong} truep: ${truep} falseN: ${falseN} falseP: ${falseP} ratioWrong: ${ratioWrong} ratioCorrect: ${ratioCorrect}")

  }
}

