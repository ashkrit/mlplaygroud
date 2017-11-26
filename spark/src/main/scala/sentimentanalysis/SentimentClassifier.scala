package sentimentanalysis

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.{SparseVector => MLVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}

object SentimentClassifier {

  def main(args: Array[String]): Unit = {

    val sparkSession = newSparkSession
    import sparkSession.implicits._

    val imdb = readFile(sparkSession, "../data/sentiment labelled sentences/imdb_labelled.txt")
    val yelp = readFile(sparkSession, "../data/sentiment labelled sentences/yelp_labelled.txt")
    val amazon = readFile(sparkSession, "../data/sentiment labelled sentences/amazon_cells_labelled.txt")

    val allData = imdb.union(yelp).union(amazon)
    val vectorModel = buildWordVectorModel(allData)


    val features = vectorModel
      .transform(allData)
      .map(vector => new LabeledPoint(vector.getDouble(1), toVector(vector.getAs(2))))

    val Array(training, test) = features.rdd.randomSplit(Array(0.7, 0.3))

    val bayesModel = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (bayesModel.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println(s"total count ${features.count()} and accuracy $accuracy")

    val df = sparkSession.createDataFrame(Seq(
      ("Highly recommend for any one who has a blue tooth phone".split(" "), 0.0d),
      ("This is crap".split(" "), 0.0d),
      ("have better things to do in life".split(" "), 0.0d),
      ("I hate you!!!!!!!!!!!!!!".split(" "), 0.0d),
      ("looks good".split(" "), 0.0d)
    )).toDF("line", "label")


    val sample = vectorModel.transform(df)
      .map(vector => new LabeledPoint(vector.getDouble(1), toVector(vector.getAs(2))))

    sample
      .map(r => bayesModel.predict(r.features))
      .foreach(a => println(a))


  }

  private def newSparkSession = {
    val sparkConf = new SparkConf().setAppName("Basic").setMaster("local")
    val sparkSession = SparkSession.builder()
      .appName("ReviewClassier")
      .config(sparkConf)
      .getOrCreate()
    sparkSession
  }

  private def readFile(sparkSession: SparkSession, fileName: String): DataFrame = {
    import sparkSession.implicits._
    sparkSession.sparkContext
      .textFile(fileName)
      .map(line => line split('\t'))
      .filter(cols => cols.length == 2 && !cols(1).isEmpty)
      .map(columns => (columns(0).split(" "), columns(1).toDouble))
      .toDF("line", "label")
  }

  private def buildWordVectorModel(data: DataFrame):CountVectorizerModel = {
    val vectorModel = new CountVectorizer()
      .setInputCol("line")
      .setOutputCol("features")
      .fit(data)
    vectorModel
  }

  def toVector(sparseVector: SparseVector): MLVector = {
    new MLVector(sparseVector.size, sparseVector.indices, sparseVector.values);
  }

}
