# Recommendation system approaches

Implementation, test and comparatives of different recommendation models base on embeddings (Collaborative filtering like).  


# Models

* **Embedding dot model**
  * Learn both users and movies embedding.
  * Dot product beethen these to predict user rattings. 
* **Embedding biases dot model**
  * Add a bias to for each user an movie. Similar to the bias in a fully-connected layer or the intercept in a linear model. It just provides an extra degree of freedom.
  * Also pass dot product output through a sigmoid layer and then scaling the result using the min and max ratings in the data. This technique introduces a non-linearity into the output and results in a loss improvement.
* **Embedding dense model**
  * Instead of perform a dot product between users and movies embedding add a fully connected layer(dense) after input embedding layers.
  * This technique is more flexible than the previous ones because allows add more feature columns as model input. 
* **User/Movie/Gender embedding dense model**
  * Same EmbeddingDenseModelFactory approach.
  * Add genders features columns.
  * Use a sigmoid layer and then scaling the result using the min and max ratings in the data.
* **Wide and deep model**
  * A mix bethween two models, linear regresion and **Embedding dense model**.
  * This model learn to combine memorization and generalization like humans do.
  * Liner model learnin to memorize.
  * Deep model generalize.
  * See: [Wide & Deep Learning: Better Together with TensorFlow](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
