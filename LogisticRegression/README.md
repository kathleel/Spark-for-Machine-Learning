
This is an example using Spark Machine Learning LogisticRegression , written in Scala, 
to demonstrate How to get started with Spark ML and Spark SQL

There are  1 datafile  in this directory :
	wbcd.csv 
 

You can run these examples in the spark shell by putting the code from the scala file in the spark shell after launching:
 
$spark-shell 

Or you can run the applications with these steps:

Step 1: mvn clean install

Step 2:

spark-submit --class lrexample1.Cancer --master yarn sparkmllr-1.0.jar


