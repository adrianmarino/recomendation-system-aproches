# Recommendation system approaches

Implementation, test and comparatives of different recommendation models (Collaborative filtering like).  
 
# Models

* **Embedding dot model**
  * Learn both users and movie embedding.
  * Dot product between these to predict user ratings.

* **Embedding biases dot model**
  * Add a bias to each user and movie. Similar to the bias in a fully-connected layer or the intercept in a linear model. It just provides an extra degree of freedom.
  * Also pass dot product output through a sigmoid layer and then scaling the result using the min and max ratings in the data. This technique introduces a non-linearity into the output and results in a loss improvement.

* **Embedding dense model**
  * Instead of performing a dot product between users and movies, embedding adds a fully connected layer(dense) after input embedding layers.
  * This technique is more flexible than the previous ones because it allows adding more feature columns as model input.

* **User/Movie/Gender embedding dense model**
  * Same **Embedding dense model** approach.
  * Add genders features columns.
  * Use a sigmoid layer and then scale the result using the min and max ratings in the data.

* **Wide and deep model**
  * A mix between two models, linear regression and **Embedding dense model**.
  * This model learns to combine memorization and generalization like humans do.
  * Liner model learning to memorize.
  * Deep model learning to generalize.
  * See: [Wide & Deep Learning: Better Together with TensorFlow](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)

* **Deep Factorization Machine Model**
  * **Pending**. See: [Deep Factorization Machine Model for CRT prediction](https://github.com/adrianmarino/deep-fm).

# Notebooks

* [Recommendation system: Approaches](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/recommendation-system-comparatives.ipynb)
* [Recommendation systems: Deep Model only](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/deep-model-user-movie.ipynb)
* User/Movie/Genders Deep Model
  * [1. Dataset preprocecing](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/1.input-data-building.ipynb)
  * [2. Dataset preprocecing and train/validation/test split](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/2.train-test-sets-building.ipynb)  
  * [3.1. Model training/validation (Pandas Ray version)](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/3.train-model-pandas-ray.ipynb)
  * [3.2. Model training/validation (Spark version)](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/3.train-model-spark.ipynb)


## Requisites

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Getting starter

**Step 1**: Clone repo.

```bash
$ git clone https://github.com/adrianmarino/recommendation-system-approaches.git
$ cd recommendation-system-approaches
```

**Step 2**: Create environment.

```bash
$ conda env create -f environment.yml
```

**Step 3**: Enable project environment.

```bash
$ conda activate recommendations
```

## Open notebooks locally

**Step 1**: Enable project environment.

```bash
$ conda activate recommendations
```

**Step 2**: Under project directory boot jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Step 3**: Go to http://localhost:8888.... as indicated in the shell output.


**Note**: Yo edit source code use [Pycharm community](https://www.jetbrains.com/pycharm/download/#section=linux) for more comfort.
