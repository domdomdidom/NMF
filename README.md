# This repo corresponds to [my Medium post!](https://medium.com/@domvandendries/engineering-your-webstore-with-nmf-a33ed5eac892?source=friends_link&sk=2f63f8b0c122a1cc24915350f407cf2c)


## Intro & Motivation
NMF is a powerful linear-algebra tool often used in recommender systems, as it performs well in sparse circumstances. NMF can extract latent "topics" from a matrix - think movie genres, article subjects, etc. It works by decomposing an *m x n* matrix into a *m x p* matrix, __W__, and a *p x n* matrix, __H__, with *p* being the number of latent topics.

![](imgs/NMF.png)

The __W__ matrix represents how much each movie-watcher/customer/author belongs to each topic. The __H__ matrix represents how much each movie/product/article belongs to each topic.


# Intrigued? Read the rest on [Medium!](https://medium.com/@domvandendries/engineering-your-webstore-with-nmf-a33ed5eac892?source=friends_link&sk=2f63f8b0c122a1cc24915350f407cf2c)
