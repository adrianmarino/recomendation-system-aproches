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


# Notebooks

* [Recommendation system: Approaches](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/recommendation-system-comparatives.ipynb)
* [Recommendation systems: Deep Model only](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/deep-model-user-movie.ipynb)
* User/movie/Genders Deep Model
  * [1. Dataset preprocecing](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/1.input-data-building.ipynb)
  * [2. Dataset preprocecing nd train, val, test split](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/2.train-test-sets-building.ipynb)  
  * [3.1. Model training/validation (pandas Ray version)](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/3.train-model-pandas-ray.ipynb)
  * [3.2. Model training/validation (Spark version)](https://github.com/adrianmarino/recommendation-system-approaches/blob/master/user-movie-genres-model/3.train-model-spark.ipynb)


