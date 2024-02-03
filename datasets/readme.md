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