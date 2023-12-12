# Catchemall : Pokemon type classification & catchability prediction application
This work is part of our final project for Le Wagon - Data Science (batch #1437)

<img src='https://i.pinimg.com/originals/d5/d5/33/d5d5333d5085402243e6c642f764f4b8.gif'  width="500">

👉 Check our app here : http://www.catchemall.fun/


## 🤖 Project Overview
Greetings, Pokémon Trainers! 🎉

We are thrilled to present My PokéApp, a powerful tool designed to enhance your Pokémon journey by providing insights beyond the capabilities of your traditional Pokédex.

Our objective is the following :

1 - Enabling the possibility for pokétrainers to take a picture of a Pokémon, and determine automatically its type based on how it looks thanks to Deep Learning

2 - Determining the catchability of a pokémon based on its statistics thanks to Machine Learning

## 📸 Deep Learning Project : Pokemon type classification (computer vision)

### Data sources
We used a dataset of 809 images from Kaggle, available [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types).

This dataset contains the first 809 pokémons, which represents the generation 1 to 7.

In addition, we scraped [Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number) in order to get a higher amount of images per type of pokémon, and we ended up with 42,000 thousand images.

### Methodology
cleaning : fonctions pour trier les images dans des dossiers par type, dupliquer les images si pokemon existant dans 2 types, split du train et test set, et preprocessing pour resizer toutes les images et leur appliquer des filtres
preprocessing : OHE pour chaque catégorie

we tried inception V3 model, vgg16, resnet

### Models & metrics
we chose resnet in the end
(description des layers dans le CNN)

### Results
40% accuracy

### Difficulties & improvements' ideas
some pokemons look like shit (example : Klefki)
too much pokemons in some types (exemple : water), hypothesis we should maybe have the same nb of images for each types next time
maybe we should have created folders for hybrid pokemons

## 🪤 Machine Learning Project : Catchability prediction (regression)

### Data sources
https://www.kaggle.com/datasets/rounakbanik/pokemon

### Methodology


### Models & metrics


### Results & improvements' ideas
