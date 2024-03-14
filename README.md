# **Social Media Data Mining Project**

## Project: Unmasking Cyberbullying: Classifying Comments and Revealing Communities
The project aims to detect toxic comments made by users on Twitter and determine if they form a community. It uses NLP and ML algorithms to classify comments and analyze user interactions to identify communities. The objective is to understand better toxic communities on social media, which helps to develop strategies to reduce their impact. The project will provide insights into the spread of toxic behavior and inform efforts to create a safer online environment. In addition, by addressing the issue of toxic comments, the project will contribute to promoting positivity and well-being on social media platforms.

## Requirements: 
- python3.6.14 or Higher. You can Download from [here](https://www.python.org/downloads/)
- Modules/libraries :
    - scikitlearn, for installation refer [here](https://scikit-learn.org/stable/install.html)
    - nltk, for installation refer [here](https://www.nltk.org/install.html)
    - pands, for installation refer [here](https://pandas.pydata.org/docs/getting_started/install.html)
    - numpy, for installation refer [here](https://numpy.org/install/)
    - networkx, for installation refer [here](https://networkx.org/documentation/stable/install.html)
    - matplotlib, for installation refer [here](https://matplotlib.org/stable/users/installing/index.html)
    - tweepy, for installation refer [here](https://github.com/tweepy/tweepy)
    - Jupyter Notebook, for installation refer [here](https://jupyter.org/install)
- System Requirements :
    - 8GB Ram (minimum)
    - 128GB of Memory (minimum)
    - Intel i7 or Mac M1 processor or higher for fast processing.
- Twitter developer access, for more details refer [here](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api)
   
### Note

It is recommended to use pip3 instead of pip, as pip will have compatability issues with python3 <br />
eg: If the below command doesn't run
```
pip install pandas
```
use this command

```
pip3 install pandas
```

## Request Code Access, to downlod it form the Github 
Run the below command to clone code from github
```
git clone https://github.com/shivakumar96/Social-Media_Data-Mining.git
```

## Running the code 
Start Jupyetr Notebook <br />
```
jupyter notebook
```
The above comand will run the Jupyter Notebook on the web browser. <br />
Now, in the Jupyter Notebook browser tab, select the folder where the code resides. <br />
<br />

Upadte you twitter developer access keys and tokens as shown below 
```
CONSUMER_KEY = "<Insert Your Key Here>"
CONSUMER_SECRET = "<Insert Your Key Here>"
OAUTH_TOKEN = "<Insert Your Key Here>"
OAUTH_TOKEN_SECRET = "<Insert Your Key Here>"
```

In the options under **cell** select **Run all** to run the entrire code <br />
or <br />
select a cell and click **Run** to execute a particular cell.

**If you want to run just as a python file, you can export the Jupyter notebook as python code, and run it using below command**
```
python3 Project.py
```
or depending on your python environment
```
python Project.py
```

## Note <br />

When it comes to executing this project, there are two methods available:  <br />

1. Using the pre-extracted dataset that has already been saved in CSV format. This method is faster since it eliminates the need to extract tweets from the Twitter API, which can take a significant amount of time due to the 15-minute delay enforced by Twitter's API policy.  <br />

2. Extracting tweets from Twitter directly and disregarding any saved datasets.  <br />


