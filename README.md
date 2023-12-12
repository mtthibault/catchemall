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

## 📸 Deep Learning project : Pokemon type classification (computer vision)

### Data sources
mettre lien Kaggle
+ expliquer : dataset avec 809 pokemons (seven generations) et une colonne type1 et une colonne type2

scraping Bulbapedia in order to get a massive amount of images and improve our accuracy

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

## 🪤 Machine Learning project : Catchability prediction (regression)

### Data sources


### Methodology


### Models & metrics


### Results & improvements' ideas
