# Catchemall : Pokemon type classification & catchability prediction application
This work is part of our final project for Le Wagon - Data Science (batch #1437)

<img src='https://i.pinimg.com/originals/d5/d5/33/d5d5333d5085402243e6c642f764f4b8.gif'  width="500">

👉 Check our app here : http://www.catchemall.fun/


## 🤖 Project Overview
Greetings, Pokémon Trainers! 🎉

We are thrilled to present My PokéApp, a powerful tool designed to enhance your Pokémon journey by providing insights beyond the capabilities of your traditional Pokédex.

Our objective is the following :

- Enabling the possibility for pokétrainers to take a picture of a pokémon, and determine automatically its type based on how it looks thanks to Deep Learning

- Determining the catchability of a pokémon based on its statistics thanks to Machine Learning

## 📸 Deep Learning Project : Pokemon type classification (computer vision)

### Data sources
We used a dataset comprising 809 images sourced from Kaggle, accessible [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types). This dataset encompasses the initial 809 pokémons, spanning generations 1 to 7, complete with their respective images and types. Notably, some pokémons possess dual types, exemplified by creatures like Moltres, a Fire and Flying type.

Furthermore, we conducted web scraping on [Pokemondb](https://pokemondb.net/) to augment our dataset, yielding a substantial collection of 42,000 images categorized by pokémon type.

### Methodology
In the preprocessing phase, we employed one-hot encoding for each pokémon type, resulting in 18 distinct categories.

Additionally, we developed functions to systematically organize each pokémon image into its corresponding type folder. This process accommodated cases where a pokémon had dual types, necessitating the duplication of its image. We also implemented functions to automate the creation of train/test datasets, with 80% of the images allocated to the training set and 20% to the testing set.

Following these preparations, we experimented with various models, including Inception V3, VGG16, and ResNet50.

### Models & metrics
After running different models, we ended up choosing ResNet50.

First, we loaded a pre-trained ResNet50 model and froze its weights to retain learned features.

A custom model was constructed by integrating data augmentation layers, the ResNet50 base model, and additional dense layers for classification.

After that, a data preprocessing function utilizing ResNet50's preprocess_input was implemented. Additionally, early stopping and learning rate reduction callbacks were set up.

In the end, the model was compiled with the Adam optimizer and categorical crossentropy loss. Training was executed on preprocessed datasets, with monitoring facilitated through early stopping and learning rate reduction.

You can explore the notebooks for a more in-depth understanding of the model architecture, data preprocessing steps, and the training process.

### Conclusion, difficulties & improvements' ideas
The model achieved a 40% accuracy on the validation dataset.

This indicates a need for improvement in the current classification approach:


- Exploring more complex model architectures and fine-tuning hyperparameters, or / an creating folders for hybrid pokémon / dual types could improve the model's ability to handle the complexities associated with dual-type classification.

- Moreover, some pokémons, like Klefki, pose challenges for the model as he doesn't have specific characteristics showing to which type he belongs to.

- Furthermore, the dataset exhibits an imbalance in the number of images per pokémon type, notably in types like Water that have much more pokémons in the Pokémon's Universe than Ghost types. To improve overall model performance, balancing the dataset by collecting more samples for underrepresented types or techniques like oversampling could lead to better results.

## 🪤 Machine Learning Project : Catchability prediction (regression)

### Data sources
https://www.kaggle.com/datasets/rounakbanik/pokemon

### Methodology


### Models & metrics


### Conclusion, difficulties & improvements' ideas
