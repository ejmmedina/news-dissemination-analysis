```pyspark
spark
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    <pyspark.sql.session.SparkSession object at 0x7f6c2babea58>


```pyspark
sc
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    <SparkContext master=yarn appName=livy-session-3>


```pyspark
sc.install_pypi_package('s3fs')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    Collecting s3fs
      Using cached https://files.pythonhosted.org/packages/72/5c/ec84c7ec49fde2c3b0d885ecae4504fa40fc77fef7684e9f2939c50f9b94/s3fs-0.4.0-py3-none-any.whl
    Collecting boto3>=1.9.91
      Using cached https://files.pythonhosted.org/packages/39/8b/250778bc5fd4b9a91d7916f8729b346931be419052130548617139247ba6/boto3-1.10.41-py2.py3-none-any.whl
    Collecting botocore>=1.12.91
      Using cached https://files.pythonhosted.org/packages/49/e0/a9a53656126a635120c0f7eaee3ff3b26a4a08596d10e7192fd91ce19e5c/botocore-1.13.41-py2.py3-none-any.whl
    Collecting fsspec>=0.6.0
      Using cached https://files.pythonhosted.org/packages/dd/1f/7028dacd3c28f34ce48130aae73a88fa5cc27b6b0e494fcf2739f7954d9d/fsspec-0.6.2-py3-none-any.whl
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/site-packages (from boto3>=1.9.91->s3fs) (0.9.4)
    Collecting s3transfer<0.3.0,>=0.2.0
      Using cached https://files.pythonhosted.org/packages/16/8a/1fc3dba0c4923c2a76e1ff0d52b305c44606da63f718d14d3231e21c51b0/s3transfer-0.2.1-py2.py3-none-any.whl
    Collecting docutils<0.16,>=0.10
      Using cached https://files.pythonhosted.org/packages/22/cd/a6aa959dca619918ccb55023b4cb151949c64d4d5d55b3f4ffd7eee0c6e8/docutils-0.15.2-py3-none-any.whl
    Collecting python-dateutil<2.8.1,>=2.1; python_version >= "2.7"
      Using cached https://files.pythonhosted.org/packages/41/17/c62faccbfbd163c7f57f3844689e3a78bae1f403648a6afb1d0866d87fbb/python_dateutil-2.8.0-py2.py3-none-any.whl
    Collecting urllib3<1.26,>=1.20; python_version >= "3.4"
      Using cached https://files.pythonhosted.org/packages/b4/40/a9837291310ee1ccc242ceb6ebfd9eb21539649f193a7c8c86ba15b98539/urllib3-1.25.7-py2.py3-none-any.whl
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/site-packages (from python-dateutil<2.8.1,>=2.1; python_version >= "2.7"->botocore>=1.12.91->s3fs) (1.12.0)
    Installing collected packages: docutils, python-dateutil, urllib3, botocore, s3transfer, boto3, fsspec, s3fs
      Found existing installation: python-dateutil 2.8.1
        Uninstalling python-dateutil-2.8.1:
          Successfully uninstalled python-dateutil-2.8.1
    Successfully installed boto3-1.10.41 botocore-1.13.41 docutils-0.15.2 fsspec-0.6.2 python-dateutil-2.8.0 s3fs-0.4.0 s3transfer-0.2.1 urllib3-1.25.7

## Reading mentions database


```pyspark
df_mention = spark.read.parquet('s3://bdcc-lab/mention.parquet')
```


```pyspark
df_mention = df_mention.drop(*['EventTimeDate', 'MentionSourceName', 'MentionIdentifier', 'InRawText', 'Confidence', 'MentionDocTranslationInfo', 'Extras'])
```

## Reading events database


```pyspark
events = spark.read.csv('s3://gdelt-open-data/v2/events/2019*.csv',
                     sep='\t',
                     header=None)
```


```pyspark
cols_events= ['GlobalEventID',
             'Day',
             'MonthYear',
             'Year',
             'FractionDate', 
             'Actor1Code',
             'Actor1Name',
             'Actor1CountryCode',
             'Actor1KnownGroupCode',
             'Actor1EthnicCode',
             'Actor1Religion1Code',
             'Actor1Religion2Code',
             'Actor1type1Code',
             'Actor1type2Code',
             'Actor1type3Code',
             'Actor2Code',
             'Actor2Name',
             'Actor2CountryCode',
             'Actor2KnownGroupCode',
             'Actor2EthnicCode',
             'Actor2Religion1Code',
             'Actor2Religion2Code',
             'Actor2type1Code',
             'Actor2type2Code',
             'Actor2type3Code',
             'IsRootEvent',
              'EventCode',
              'EventBaseCode',
              'EventRootCode',
              'QuadClass',
              'GoldsteinScale',
              'NumMentions',
              'NumSources',
              'NumArticles',
              'AvgTone',
              'Actor1Geo_Type',
              'Actor1Geo_Fullname',
              'Actor1Geo_CountryCode',
              'Actor1Geo_ADM1Code',
              'Actor1Geo_ADM2Code',
              'Actor1Geo_Lat',
              'Actor1Geo_Long',
              'Actor1Geo_FeatureID',
              'Actor2Geo_Type',
              'Actor2Geo_Fullname',
              'Actor2Geo_CountryCode',
              'Actor2Geo_ADM1Code',
              'Actor2Geo_ADM2Code',
              'Actor2Geo_Lat',
              'Actor2Geo_Long',
              'Actor2Geo_FeatureID',
              'ActionGeo_Type',
              'ActionGeo_Fullname',
              'ActionGeo_CountryCode',
              'ActionGeo_ADM1Code',
              'ActionGeo_ADM2Code',
              'ActionGeo_Lat',
              'ActionGeo_Long',
              'ActionGeo_FeatureID',
              'DATEADDED',
              'SOURCEURL'
             ]

for i, col_names in enumerate(cols_events):
    events = events.withColumnRenamed('_c'+str(i), col_names)
```


```pyspark
events.write.parquet('s3://bdcc-lab/all_events.parquet')
```


```pyspark
events.createOrReplaceTempView('events')
```

### Filtering to Philippines only


```pyspark
phil_events = spark.sql('''select * from events 
            where Actor1CountryCode == "RP" or Actor1Geo_CountryCode == "RP" or Actor2CountryCode == "RP" or 
            Actor2Geo_CountryCode == "RP" or ActionGeo_CountryCode == "RP"
            ''')
```


```pyspark
phil_events = phil_events.withColumn('mention_source', phil_events['NumMentions']/phil_events['NumSources'])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
phil_events = phil_events.withColumn('mention_article', phil_events['NumMentions']/phil_events['NumArticles'])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
phil_events.write.parquet('s3://bdcc-lab/phil_events_final.parquet')
```


```pyspark
phil_events = phil_events[[
    'GlobalEventID',
         'Day',
         'Actor1Code',
         'Actor1CountryCode',
         'Actor1KnownGroupCode',
         'Actor1EthnicCode',
         'Actor1Religion1Code',
         'Actor1Religion2Code',
         'Actor1type1Code',
         'Actor1type2Code',
         'Actor1type3Code',
         'Actor2Code',
         'Actor2CountryCode',
         'Actor2KnownGroupCode',
         'Actor2EthnicCode',
         'Actor2Religion1Code',
         'Actor2Religion2Code',
         'Actor2type1Code',
         'Actor2type2Code',
         'Actor2type3Code',
         'IsRootEvent',
          'EventCode', 
        'EventBaseCode',
        'EventRootCode',
         'QuadClass',
         'NumMentions',
         'NumSources',
         'NumArticles',
        'AvgTone'
]]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
phil_events.write.parquet('s3://bdcc-lab/phil_events_final.parquet')
```

## Reading mentions parquet dataset


```pyspark
df_mention = spark.read.parquet('s3://bdcc-lab/mention.parquet')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_mention = df_mention.drop(*['EventTimeDate', 'MentionSourceName', 'MentionIdentifier', 'InRawText', 'Confidence', 'MentionDocTranslationInfo', 'Extras'])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


## Reading philippine events parquet dataset


```pyspark
phil_events = spark.read.parquet('s3://bdcc-lab/phil_events_final.parquet')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


## Filtering the features


```pyspark
phil_events = phil_events[[
    'GlobalEventID',
         'Day',
         'Actor1Code',
         'Actor1CountryCode',
         'Actor1KnownGroupCode',
         'Actor1EthnicCode',
         'Actor1Religion1Code',
         'Actor1Religion2Code',
         'Actor1type1Code',
         'Actor1type2Code',
         'Actor1type3Code',
         'Actor2Code',
         'Actor2CountryCode',
         'Actor2KnownGroupCode',
         'Actor2EthnicCode',
         'Actor2Religion1Code',
         'Actor2Religion2Code',
         'Actor2type1Code',
         'Actor2type2Code',
         'Actor2type3Code',
         'IsRootEvent',
          'EventCode', 
        'EventBaseCode',
        'EventRootCode',
         'QuadClass',
         'NumMentions',
         'NumSources',
         'NumArticles',
        'AvgTone'
]]
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


## Feature engineering


```pyspark
phil_events = phil_events.withColumn('mention_source', phil_events['NumMentions']/phil_events['NumSources'])
phil_events = phil_events.withColumn('mention_article', phil_events['NumMentions']/phil_events['NumArticles'])
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
phil_events.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- GlobalEventID: string (nullable = true)
     |-- Day: string (nullable = true)
     |-- Actor1Code: string (nullable = true)
     |-- Actor1CountryCode: string (nullable = true)
     |-- Actor1KnownGroupCode: string (nullable = true)
     |-- Actor1EthnicCode: string (nullable = true)
     |-- Actor1Religion1Code: string (nullable = true)
     |-- Actor1Religion2Code: string (nullable = true)
     |-- Actor1type1Code: string (nullable = true)
     |-- Actor1type2Code: string (nullable = true)
     |-- Actor1type3Code: string (nullable = true)
     |-- Actor2Code: string (nullable = true)
     |-- Actor2CountryCode: string (nullable = true)
     |-- Actor2KnownGroupCode: string (nullable = true)
     |-- Actor2EthnicCode: string (nullable = true)
     |-- Actor2Religion1Code: string (nullable = true)
     |-- Actor2Religion2Code: string (nullable = true)
     |-- Actor2type1Code: string (nullable = true)
     |-- Actor2type2Code: string (nullable = true)
     |-- Actor2type3Code: string (nullable = true)
     |-- IsRootEvent: string (nullable = true)
     |-- EventCode: string (nullable = true)
     |-- EventBaseCode: string (nullable = true)
     |-- EventRootCode: string (nullable = true)
     |-- QuadClass: string (nullable = true)
     |-- NumMentions: string (nullable = true)
     |-- NumSources: string (nullable = true)
     |-- NumArticles: string (nullable = true)
     |-- AvgTone: string (nullable = true)
     |-- mention_source: double (nullable = true)
     |-- mention_article: double (nullable = true)


```pyspark
phil_events.show()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    +-------------+--------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+-----------+---------+-------------+-------------+---------+-----------+----------+-----------+-----------------+--------------+---------------+
    |GlobalEventID|     Day|Actor1Code|Actor1CountryCode|Actor1KnownGroupCode|Actor1EthnicCode|Actor1Religion1Code|Actor1Religion2Code|Actor1type1Code|Actor1type2Code|Actor1type3Code|Actor2Code|Actor2CountryCode|Actor2KnownGroupCode|Actor2EthnicCode|Actor2Religion1Code|Actor2Religion2Code|Actor2type1Code|Actor2type2Code|Actor2type3Code|IsRootEvent|EventCode|EventBaseCode|EventRootCode|QuadClass|NumMentions|NumSources|NumArticles|          AvgTone|mention_source|mention_article|
    +-------------+--------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+-----------+---------+-------------+-------------+---------+-----------+----------+-----------+-----------------+--------------+---------------+
    |    820546778|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       GOV|             null|                null|            null|               null|               null|            GOV|           null|           null|          1|      020|          020|           02|        1|          2|         1|          2|-2.01005025125628|           2.0|            1.0|
    |    820546819|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          1|      043|          043|           04|        1|          5|         1|          5|-1.75438596491228|           5.0|            1.0|
    |    820546820|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          1|      043|          043|           04|        1|          2|         1|          2|-1.75438596491228|           2.0|            1.0|
    |    820546822|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          0|      061|          061|           06|        2|          3|         1|          3|-1.75438596491228|           3.0|            1.0|
    |    820546823|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          0|      061|          061|           06|        2|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820546824|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|    PHLGOV|              PHL|                null|            null|               null|               null|            GOV|           null|           null|          1|      020|          020|           02|        1|          6|         1|          6|-2.01005025125628|           6.0|            1.0|
    |    820546833|20190201|      null|             null|                null|            null|               null|               null|           null|           null|           null|       UAF|             null|                null|            null|               null|               null|            UAF|           null|           null|          0|     1123|          112|           11|        3|          3|         1|          3|-5.96107055961071|           3.0|            1.0|
    |    820547111|20190201|    CHRCTH|             null|                null|            null|                CHR|                CTH|           null|           null|           null|       MOS|             null|                null|            null|                MOS|               null|           null|           null|           null|          0|      057|          057|           05|        1|          2|         1|          2|-5.96107055961071|           2.0|            1.0|
    |    820547112|20190201|    CHRCTH|             null|                null|            null|                CHR|                CTH|           null|           null|           null|       MOS|             null|                null|            null|                MOS|               null|           null|           null|           null|          0|      057|          057|           05|        1|          3|         1|          3|-5.96107055961071|           3.0|            1.0|
    |    820547198|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|      null|             null|                null|            null|               null|               null|           null|           null|           null|          0|      111|          111|           11|        3|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547200|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       BUS|             null|                null|            null|               null|               null|            BUS|           null|           null|          0|      051|          051|           05|        1|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547201|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       BUS|             null|                null|            null|               null|               null|            BUS|           null|           null|          0|      051|          051|           05|        1|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547203|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       BUS|             null|                null|            null|               null|               null|            BUS|           null|           null|          0|      111|          111|           11|        3|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547204|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       BUS|             null|                null|            null|               null|               null|            BUS|           null|           null|          0|      111|          111|           11|        3|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547205|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       ELI|             null|                null|            null|               null|               null|            ELI|           null|           null|          0|      061|          061|           06|        2|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547206|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          0|      061|          061|           06|        2|          5|         1|          5|-1.75438596491228|           5.0|            1.0|
    |    820547207|20190201|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|       PHL|              PHL|                null|            null|               null|               null|           null|           null|           null|          0|      061|          061|           06|        2|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547298|20190201|       ELI|             null|                null|            null|               null|               null|            ELI|           null|           null|       CUB|              CUB|                null|            null|               null|               null|           null|           null|           null|          0|      061|          061|           06|        2|          1|         1|          1|-1.75438596491228|           1.0|            1.0|
    |    820547363|20190201|       GOV|             null|                null|            null|               null|               null|            GOV|           null|           null|      null|             null|                null|            null|               null|               null|           null|           null|           null|          1|      013|          013|           01|        1|          6|         1|          6| 0.33222591362126|           6.0|            1.0|
    |    820547385|20190201|       GOV|             null|                null|            null|               null|               null|            GOV|           null|           null|      null|             null|                null|            null|               null|               null|           null|           null|           null|          1|      130|          130|           13|        3|         10|         1|         10|-4.54545454545455|          10.0|            1.0|
    +-------------+--------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+----------+-----------------+--------------------+----------------+-------------------+-------------------+---------------+---------------+---------------+-----------+---------+-------------+-------------+---------+-----------+----------+-----------+-----------------+--------------+---------------+
    only showing top 20 rows

## Merging files


```pyspark
#mentions_count = 9250645
#phile_events_count = 195826
#innerjoin = 44069
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
sample_joined = df_mention.join(phil_events, "GlobalEventID")
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
sample_joined.count()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    44069


```pyspark
sample_joined.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- GlobalEventID: string (nullable = true)
     |-- MentionTimeDate: string (nullable = true)
     |-- MentionType: string (nullable = true)
     |-- SentenceID: string (nullable = true)
     |-- Actor1CharOffset: string (nullable = true)
     |-- Actor2CharOffset: string (nullable = true)
     |-- ActionCharOffset: string (nullable = true)
     |-- MentionDocLen: string (nullable = true)
     |-- MentionDocTone: string (nullable = true)
     |-- Day: string (nullable = true)
     |-- Actor1Code: string (nullable = true)
     |-- Actor1CountryCode: string (nullable = true)
     |-- Actor1KnownGroupCode: string (nullable = true)
     |-- Actor1EthnicCode: string (nullable = true)
     |-- Actor1Religion1Code: string (nullable = true)
     |-- Actor1Religion2Code: string (nullable = true)
     |-- Actor1type1Code: string (nullable = true)
     |-- Actor1type2Code: string (nullable = true)
     |-- Actor1type3Code: string (nullable = true)
     |-- Actor2Code: string (nullable = true)
     |-- Actor2CountryCode: string (nullable = true)
     |-- Actor2KnownGroupCode: string (nullable = true)
     |-- Actor2EthnicCode: string (nullable = true)
     |-- Actor2Religion1Code: string (nullable = true)
     |-- Actor2Religion2Code: string (nullable = true)
     |-- Actor2type1Code: string (nullable = true)
     |-- Actor2type2Code: string (nullable = true)
     |-- Actor2type3Code: string (nullable = true)
     |-- IsRootEvent: string (nullable = true)
     |-- EventCode: string (nullable = true)
     |-- EventBaseCode: string (nullable = true)
     |-- EventRootCode: string (nullable = true)
     |-- QuadClass: string (nullable = true)
     |-- NumMentions: string (nullable = true)
     |-- NumSources: string (nullable = true)
     |-- NumArticles: string (nullable = true)
     |-- AvgTone: string (nullable = true)
     |-- mention_source: double (nullable = true)
     |-- mention_article: double (nullable = true)


```pyspark
sample_joined.write.parquet('s3://bdcc-lab/joined_table.parquet')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_merged = sample_joined.toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_merged.isnull().sum()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    GlobalEventID               0
    MentionTimeDate             0
    MentionType                 0
    SentenceID                  0
    Actor1CharOffset            0
    Actor2CharOffset            0
    ActionCharOffset            0
    MentionDocLen               0
    MentionDocTone              0
    Day                         0
    Actor1Code               5244
    Actor1CountryCode       27066
    Actor1KnownGroupCode    43549
    Actor1EthnicCode        43911
    Actor1Religion1Code     42504
    Actor1Religion2Code     43752
    Actor1type1Code         20629
    Actor1type2Code         42659
    Actor1type3Code         44035
    Actor2Code              16620
    Actor2CountryCode       32768
    Actor2KnownGroupCode    43524
    Actor2EthnicCode        43955
    Actor2Religion1Code     41575
    Actor2Religion2Code     42561
    Actor2type1Code         28437
    Actor2type2Code         43012
    Actor2type3Code         44058
    IsRootEvent                 0
    EventCode                   0
    EventBaseCode               0
    EventRootCode               0
    QuadClass                   0
    NumMentions                 0
    NumSources                  0
    NumArticles                 0
    AvgTone                     0
    mention_source              0
    mention_article             0
    dtype: int64


```pyspark
df_merged.to_parquet('s3://bdcc-lab/df_merged.parquet')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_merged.to_pickle('df_merged.pkl')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


      GlobalEventID MentionTimeDate  ... mention_source mention_article
    0     813443217  20190101081500  ...           10.0             1.0
    1     813443217  20190101080000  ...           10.0             1.0
    2     813443217  20190101040000  ...           10.0             1.0
    3     813449194  20190101053000  ...           10.0             1.0
    4     813494031  20190101150000  ...           10.0             1.0
    
    [5 rows x 39 columns]

## Reading merged parquet file


```pyspark
df_merged = spark.read.parquet('s3://bdcc-lab/df_merged.parquet')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_merged.printSchema()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…


    root
     |-- GlobalEventID: string (nullable = true)
     |-- MentionTimeDate: string (nullable = true)
     |-- MentionType: string (nullable = true)
     |-- SentenceID: string (nullable = true)
     |-- Actor1CharOffset: string (nullable = true)
     |-- Actor2CharOffset: string (nullable = true)
     |-- ActionCharOffset: string (nullable = true)
     |-- MentionDocLen: string (nullable = true)
     |-- MentionDocTone: string (nullable = true)
     |-- Day: string (nullable = true)
     |-- Actor1Code: string (nullable = true)
     |-- Actor1CountryCode: string (nullable = true)
     |-- Actor1KnownGroupCode: string (nullable = true)
     |-- Actor1EthnicCode: string (nullable = true)
     |-- Actor1Religion1Code: string (nullable = true)
     |-- Actor1Religion2Code: string (nullable = true)
     |-- Actor1type1Code: string (nullable = true)
     |-- Actor1type2Code: string (nullable = true)
     |-- Actor1type3Code: string (nullable = true)
     |-- Actor2Code: string (nullable = true)
     |-- Actor2CountryCode: string (nullable = true)
     |-- Actor2KnownGroupCode: string (nullable = true)
     |-- Actor2EthnicCode: string (nullable = true)
     |-- Actor2Religion1Code: string (nullable = true)
     |-- Actor2Religion2Code: string (nullable = true)
     |-- Actor2type1Code: string (nullable = true)
     |-- Actor2type2Code: string (nullable = true)
     |-- Actor2type3Code: string (nullable = true)
     |-- IsRootEvent: string (nullable = true)
     |-- EventCode: string (nullable = true)
     |-- EventBaseCode: string (nullable = true)
     |-- EventRootCode: string (nullable = true)
     |-- QuadClass: string (nullable = true)
     |-- NumMentions: string (nullable = true)
     |-- NumSources: string (nullable = true)
     |-- NumArticles: string (nullable = true)
     |-- AvgTone: string (nullable = true)
     |-- mention_source: double (nullable = true)
     |-- mention_article: double (nullable = true)


```pyspark
df_merged = df_merged.toPandas()
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…



```pyspark
df_merged.to_csv('s3://bdcc-lab/bdcc_lab3_data.csv')
```


    VBox()



    FloatProgress(value=0.0, bar_style='info', description='Progress:', layout=Layout(height='25px', width='50%'),…

