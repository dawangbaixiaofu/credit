## download
download data from kaggle using cml in windows system.
```sh
# kaggle 
# install and authentication
pip install kaggle
# kaggle's Account--user profile -- Create New Token
# then download a json file named kaggle.json containing API credentials
# move kaggle.json file to location: C:\Users\<Windows-username>\.kaggle\kaggle.json
# if use a kaggle API, there is no need to move 

# note: before download data either using api or cml, must accept rules, but don't via api or cml\
# do these by visiting kaggle website and accept the rules there.

# CLI (https://github.com/Kaggle/kaggle-api#competitions)
# some commands about competitions
kaggle competitions list :list the currently active competitions
kaggle competitions download -c [COMPETITION]: download files associated with a competition
kaggle competitions submit -c [COMPETITION] -f [FILE] -m [MESSAGE]: make a competition submission

# some commands about Datasets(https://github.com/Kaggle/kaggle-api#datasets)
kaggle datasets list -s [KEYWORD]: list datasets matching a search term
kaggle datasets download -d [DATASET]: download files associated with a dataset
```

## data distribution 
统计label的分布，正负样本比例是否均衡。如果不均衡训练模型的时候要考虑使用上采样的方式进行训练。
统计各个feature的分布。1.看各个特征数据是否呈现有长尾分布的情况；2.异常值是否存在有多少；

## MySQL
使用Docker的MySQL镜像快速的开启一个容器，使用dataframe.to_sql方法把数据存入到数据库中。


## data encoder configuration 
根据特征统计的数据分布，配置各个特征的离散化的切分点。
give me some credit项目特征的数据类型是数值型，所以切分配置较为简单。
最后把设置好的特征编码配置存入表中。