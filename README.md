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
We used a dataset comprising 809 images sourced from Kaggle, accessible [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types). This dataset encompasses the initial 809 Pokémon, spanning generations 1 to 7, complete with their respective images and types. Notably, certain Pokémon possess dual types, exemplified by creatures like Moltres, a Fire and Flying type.

Furthermore, we conducted web scraping on [Bulbapedia](https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number) to augment our dataset, yielding a substantial collection of 42,000 images categorized by Pokémon type.

### Methodology
In the preprocessing phase, we employed one-hot encoding for each Pokémon type, resulting in 18 distinct categories.

Additionally, we developed functions to systematically organize each Pokémon image into its corresponding type folder. This process accommodated cases where a Pokémon had dual types, necessitating the duplication of its image. We also implemented functions to automate the creation of train/test datasets, with 80% of the images allocated to the training set and 20% to the testing set.

Following these preparations, we experimented with various models, including Inception V3, VGG16, and ResNet50.

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
