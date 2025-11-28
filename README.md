# SpotifyRecommender
Engine that finds spotify songs most like your favorite songs using NLP

## ℹ️ Overview
The script uses Spotify's official API in order to scrape past, play, or analyze user's most recent or liked tracks.
Using this info, an NLP model will analyze your tastes as compared to a datsaset of 110k+ songs and will 

## 📚 Techstack
> *It's important to note that you don't need to know how to code to use this bot effectively*

Languages used are Python and PostgreSQL. 
For data collection, we store all the data remotely using a PostgreSQL server hosted on AWS. 
Libraries are included in the requirements.txt

## 📋 Requirements
A good understanding of PyTorch, as well as python/sql will be required if you want to contribute/develop this further.

To use the bot, you will need a Spotify Developer's account and will need a client ID and client SECRET if you want the code to control your songs entirely. Otherwise, the model will solely work based on songs you input into it (the model will become more accurate if you're able to feed it a couple of songs). 
