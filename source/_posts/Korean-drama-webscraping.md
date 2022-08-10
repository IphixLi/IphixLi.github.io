---
title: Korean drama webscraping
date: 2022-08-09 22:10:11
tags:
- webscraping
- data engineering
- data mining
- Introduction
- k-drama
- BeautifulSoup
- Pandas
categories:
- Projects
toc: true
cover: /gallery/hosplaylist.jpg
thumbnail: /gallery/hosplaylist.jpg
---
<p class="not-gallery-item" style="font-size:13px" align="center"><em>(on the cover) Hospital Playlist</em></p>

<em>TLDR: I made a list of korean Dramas with rankings and other cool descriptors for each drama</em>

## Introduction

As an avid Korean drama watcher, finding the next great drama to watch is the real deal.  From my personal experience, most Korean dramas plots are slow-burn meaning that you have to watch multiple episode to understand well the story. For shows which takes more than 10 50-minute-ish episodes, getting a good drama recommendation is a life-saver. 

When we had a webscraping class during my intro to data science class, I quickly grasped my passion project:Building a giant list of Korean dramas from different websites and then analyse which drama to watch and where.

## Process

I collected ratings, platforms, release year, number of episodes, tags,airing times, episode length, description and many more descriptors with the goal of developing robust recommender system for myself.

I used Beautifulsoup python library to webscrape [Wikipedia](https://en.wikipedia.org/wiki/List_of_South_Korean_dramas), [mydramalist](https://mydramalist.com/), and [imdb](https://www.imdb.com/) websites.

MyDramalist is a website containing a  variety of asian dramas and movies with excellent breadth when it comes to giving drama metadata. The Internet Movie Database (IMDb) is an online database containing information and statistics about movies, TV shows and video games as well as actors, directors and other film industry professionals. Its popularity means it gives credible ratings as it have large user base.

## Planning

I webscraped Korean drama list from wikipedia gaining about 1500 korean dramas.

```python
from bs4 import BeautifulSoup
import requests

def get_wikilinks():
    url="https://en.wikipedia.org/wiki/List_of_South_Korean_dramas"
    r=requests.get(url)
    soup=BeautifulSoup(r.content,"html.parser",from_encoding='utf-8')

    x=soup.find_all('ul')
    start=0
    movies={}
    for i in x:
        if start>=2:
            break
        for a in i.find_all('a', href=True):
            if a['href']=="#See_also":
                start+=1
            elif a['href']=="/wiki/List_of_South_Korean_television_series":
                start+=1
                break
            elif start:
                movies[a.get_text().strip()]="https://en.wikipedia.org"+a['href']
    return movies
```
<em>code for scrapping list of korean dramas from wikipedia</em>

 I then used the list to find a breadth of descriptors from mydramalist website through search and scrape strategy getting about 1300 korean dramas. 

On the side, I webscraped IMDB to get imdb rating and concise drama description, generating about 1510 korean drama. I finally joined one from dramalist and imdb, leading to a final list of about 750 korean drama, this was because imdb list included miniseries or korean drama names differed drastically from names in mydramalist.

```python
# getting dramalist from wikipedia
wikilinks=get_wikilinks()
dramas=wikilinks.keys()

##test example (it is important to keep a small list for testing because websites limit number
# of requests so limiting webscraping instances is crucial)
test=['The Greatest Marriage',"Vincenzo"]

#get mydramalist korean dramalist
dramalist=get_dramalist(dramas)
imdb=get_imdb_ratings()[0]

#contains 'Also Known as' section of mydramalist dramalist
other_names=dramalist[1]

#store other descriptors from mydramalist
drama_descr=dramalist[0]

#stores final list of dramas
with_imdb={}
for movie in imdb.keys():
    for drama in drama_descr.keys():
        #check if drama name from imdb is same as name in mydramalist name or is in
        # 'Also known as' section of mydramalist descriptors
        if drama in other_names.keys() and (movie.lower()==drama.lower() or (movie.lower() in other_names[drama])):
            with_imdb[drama]=drama_descr[drama]
            with_imdb[drama]['imdb_name']=movie
            with_imdb[drama]['imdb_rating']=imdb[movie][0]
            with_imdb[drama]['imdb_user_count']=imdb[movie][1]
            with_imdb[drama]['imdb_description']=imdb[movie][2]
            ###(testing outputs ongo helps to test data sanity)
            #print([drama,with_imdb[drama]])
for val in with_imdb:
    #formatting entries
    with_imdb[val]["Tags"][-1]=with_imdb[val]["Tags"][-1].replace("(vote or add tags)","")
    temp=re.findall("\d+\.?\d*",with_imdb[drama]["imdb_rating"])
    if len(temp)==1:
        with_imdb[drama]["imdb_rating"]=temp[0]

df=pd.DataFrame(with_imdb)
df = df.transpose()
###storing korean drama list into a pandas dataframe
df.to_csv("kdramalist.csv", encoding='utf-8', index=False,na_rep="N/A")

###evaluate accuracy
print("#############lengths: ",(len(with_imdb),len(drama_descr)))
#############lengths:  (741, 1279)

```
this dataset can be used for multiple data science projects including sentiment analysis,supervised learning and recommender systems.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/iphigeniebera/korean-drama-list-about-740-unique-dramas) and and look at the full code from [Github](https://github.com/IphixLi/kdrama_webscraping).

********************************************************************************************************************
<em>This Webscraping project is part of Kdrama project in which I practice my data science and engineering skills, step by step until I achieve highly accurate model for recommending korean drama.</em>

<strong>My milestone projects (not in order) are as below:</strong>
1. Data collection
	- Webscraping (done) (read the write above on webscraping project)
	- APIs 
	- Datasets
2. ETL process:the goal is to automate data collection
	- data modelling
    - Data pipelining
    - Dashboarding
4. Data visualization and statistics
	- metric analysis
	- more Dashboards
	- visualization
3. Machine learning:(the cool part, you know)
	- machine learning models
	- Recommender system :)
	- Sentiment analysis and NLP
4. Advanced
	- sophisticated machine learning
	- analytics using alternate data and AI
	- computing advancements




