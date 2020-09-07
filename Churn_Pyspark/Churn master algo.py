#/usr/hdp/current/spark2-client/bin/pyspark
import numpy as np 
import pandas as pd
from pyspark import SparkConf, SparkContext, SQLContext
import pyspark
import functools
import sys
import os
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation



##########################   SAMPLE DATA AND SELECT RELEVANT COLUMNS ACCRODING TO IV AND MULTICOLLINEARITY RESULTS ################################
data=df.sample(False, 0.20, seed=0)
##########################   SAMPLE DATA AND SELECT RELEVANT COLUMNS ACCRODING TO IV AND MULTICOLLINEARITY RESULTS ################################



############################################# PREPARE DATA FOR TRAINING #######################################################################
ignore = ['CID', 'CHURN_FLAG'] 
feature = VectorAssembler(inputCols=[x for x in data.columns if x not in ignore],outputCol='features')
feature_vector= feature.transform(data)
(trainingData, testData) = feature_vector.randomSplit([0.7, 0.3],seed = 11)
trainingData = trainingData.persist(pyspark.StorageLevel.DISK_ONLY)
testData = testData.persist(pyspark.StorageLevel.DISK_ONLY)
############################################# PREPARE DATA FOR TRAINING #######################################################################



############################################# LOGISTIC REGRESSION #######################################################################
lr = LogisticRegression(labelCol="CHURN_FLAG", featuresCol="features")
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
#TIME : 20 min

accuracy = MulticlassClassificationEvaluator(labelCol="CHURN_FLAG", predictionCol="prediction", metricName="accuracy")
roc = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='CHURN_FLAG',metricName="areaUnderROC") 
lr_acc = accuracy.evaluate(lr_prediction)
lr_roc = ROC.evaluate(lr_prediction)
print('A LR Tree algorithm had an auc of: {0:2.2f}%'.format(lr_roc*100)) #77.6%
print('A LR algorithm had an accuracy of: {0:2.2f}%'.format(lr_acc*100))  #76.27
#gross_sales > 20 filter does increase accuracy by 1%


tp = lr_prediction[(lr_prediction.CHURN_FLAG == 1) & (lr_prediction.prediction == 1)].count()
tn = lr_prediction[(lr_prediction.CHURN_FLAG == 0) & (lr_prediction.prediction == 0)].count()
fp = lr_prediction[(lr_prediction.CHURN_FLAG == 0) & (lr_prediction.prediction == 1)].count()
fn = lr_prediction[(lr_prediction.CHURN_FLAG == 1) & (lr_prediction.prediction == 0)].count()
print "True Positives:", tp #1,97,232
print "True Negatives:", tn  #3,08,596
print "False Positives:", fp  #72,484
print "False Negatives:", fn  #75494
print "Total", df.count()
r = float(tp)/(tp + fn)
print "recall", r  #0.723187374874
p = float(tp) / (tp + fp)
print "precision", p  #0.731258064038

# ------------------------------------------------------------ TUNINING (RUNS FOR A WHILE) ---------------------------------------------------------------#
evaluator = BinaryClassificationEvaluator(labelCol="CHURN_FLAG")
paramGrid = ParamGridBuilder()\
    .addGrid(lr.aggregationDepth,[2,5,10])\
    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
    .addGrid(lr.fitIntercept,[False, True])\
    .addGrid(lr.maxIter,[10, 100, 1000])\
    .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
    .build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(trainingData)
predict_train=cvModel.transform(trainingData)
predict_test=cvModel.transform(testData)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))

# ------------------------------------------------------------ TUNINING ---------------------------------------------------------------#
############################################# LOGISTIC REGRESSION #######################################################################


#################################################  RANDOM FOREST ######################################################################
rf = RandomForestClassifier(labelCol="CHURN_FLAG", featuresCol="features")
rf_model = rf.fit(trainingData)
feature_importance = rf_model.featureImportances
rf_prediction = rf_model.transform(testData)

tp = rf_prediction[(rf_prediction.CHURN_FLAG == 1) & (rf_prediction.prediction == 1)].count()
tn = rf_prediction[(rf_prediction.CHURN_FLAG == 0) & (rf_prediction.prediction == 0)].count()
fp = rf_prediction[(rf_prediction.CHURN_FLAG == 0) & (rf_prediction.prediction == 1)].count()
fn = rf_prediction[(rf_prediction.CHURN_FLAG == 1) & (rf_prediction.prediction == 0)].count()
print "True Positives:", tp  #200278
print "True Negatives:", tn #299045
print "False Positives:", fp #81805
print "False Negatives:", fn #72678
print "Total", df.count()
r = float(tp)/(tp + fn)
print "recall", r
p = float(tp) / (tp + fp)
print "precision", p


accuracy = MulticlassClassificationEvaluator(labelCol="CHURN_FLAG", predictionCol="prediction", metricName="accuracy")
roc = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='CHURN_FLAG',metricName="areaUnderROC") 
rf_acc = accuracy.evaluate(rf_prediction)
rf_roc = roc.evaluate(rf_prediction)
print('A rf Tree algorithm had an accuarcy of: {0:2.2f}%'.format(rf_acc*100)) #76.31
print('A rf algorithm had an roc of: {0:2.2f}%'.format(rf_roc*100))  #75.17

#accuracy when all variables: more or less the same

tp = rf_prediction[(rf_prediction.CHURN_FLAG == 1) & (rf_prediction.prediction == 1)].count()
tn = rf_prediction[(rf_prediction.CHURN_FLAG == 0) & (rf_prediction.prediction == 0)].count()
fp = rf_prediction[(rf_prediction.CHURN_FLAG == 0) & (rf_prediction.prediction == 1)].count()
fn = rf_prediction[(rf_prediction.CHURN_FLAG == 1) & (rf_prediction.prediction == 0)].count()
print "True Positives:", tp 
print "True Negatives:", tn  
print "False Positives:", fp  
print "False Negatives:", fn  

r = float(tp)/(tp + fn)
print "recall", r  #0.69221507054
p = float(tp) / (tp + fp)
print "precision", p  #0.71280468966
#################################################  RANDOM FOREST ######################################################################

#################################################################  GBM  ###############################################################
gbt = GBTClassifier(labelCol="CHURN_FLAG",featuresCol="features", maxIter=10)
#paramGrid = (ParamGridBuilder()
#             .addGrid(gbt.maxDepth, [2, 4, 6])
#             .addGrid(gbt.maxBins, [20, 60])
#             .addGrid(gbt.maxIter, [10, 20])
#             .build())		 
evaluator = BinaryClassificationEvaluator(labelCol="CHURN_FLAG")
paramGrid = ParamGridBuilder().build()
crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=6)
model = crossval.fit(trainingData)
model.save(sc, "gbt.model")
print model.featureImportances
gbt_prediction = model.transform(testData) 
result_df = gbt_prediction.sample(False, 0.01, seed=0)
result_df.select("*").toPandas()

#Output of result_df
6631   [0.6371655765140758, 0.3628344234859242]         0.0
6632   [0.5955572664461132, 0.4044427335538868]         0.0
6633  [0.5437952718016075, 0.45620472819839253]         0.0
6634  [0.33303150977362445, 0.6669684902263755]         1.0
6635   [0.3742899620878452, 0.6257100379121547]         1.0
#Output of result_df


#TIME : 1HR
accuracy = MulticlassClassificationEvaluator(labelCol="CHURN_FLAG", predictionCol="prediction", metricName="accuracy")
roc = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='CHURN_FLAG',metricName="areaUnderROC")
gb_acc = accuracy.evaluate(gbt_prediction)
gb_roc = roc.evaluate(gbt_prediction)

print('A gbm algorithm had an accuracy of: {0:2.2f}%'.format(gb_acc*100))  
print('A gbm algorithm had an roc of: {0:2.2f}%'.format(gb_roc*100)) 

tp = gbt_prediction[(gbt_prediction.CHURN_FLAG == 1) & (gbt_prediction.prediction == 1)].count()
tn = gbt_prediction[(gbt_prediction.CHURN_FLAG == 0) & (gbt_prediction.prediction == 0)].count()
fp = gbt_prediction[(gbt_prediction.CHURN_FLAG == 0) & (gbt_prediction.prediction == 1)].count()
fn = gbt_prediction[(gbt_prediction.CHURN_FLAG == 1) & (gbt_prediction.prediction == 0)].count()
print "True Positives:", tp  #200,921
print "True Negatives:", tn #310,523
print "False Positives:", fp #703,27
print "False Negatives:", fn ##720,35
print "Total", df.count()
r = float(tp)/(tp + fn)
print "recall", r
p = float(tp) / (tp + fp)
print "precision", p
#################################################################  GBM  ###############################################################

######################### featureImportances ################################################
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

ExtractFeatureImp(rf_model.featureImportances, trainingData, "features").head(100)
ExtractFeatureImp(gbt.featureImportances, trainingData, "features").head(100)
ExtractFeatureImp(lr_model.featureImportances, trainingData, "features").head(100)
######################### featureImportances ################################################


