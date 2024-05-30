# Music Genre Classification with Machine Learning

Using neural networks to classify music based on genre.

---

Please refer to the CS 489 Music Genre Classification.ipynb notebook to see the entire project documentation and explanation. 


# Neural Networks

In recent years, Convolutional Neural Networks (CNNs) have been lauded for their effectiveness at image classification and recognition, and they are also increasingly being applied to music genre analysis. These CNNs often times use MFCCs and Mel-spectrograms, which are essentially images - to classify audio snippets into genres. They have shown up as the clear winner compared to other classifiers, with accuracy rates among the high 80s to low 90s amongst ten different genres. Due to its newfound appeal and its compelling results, my project will be focused on using CNNs for music genre analysis.

I will first implement a Multi-layer Perceptron Neural Network because it will help introduce me to neural networks as a whole. It is considered the simpler, less effective predecessor of CNNs. I will then implement a Convolutional Neural Network (CNN). I would like to see how both approaches compare with regards to music genre classification.

The feature that I will use to train my models is Mel-frequency cepstral coefficients (MFCCs). For data, I will be using the [GZTAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download) provided by Kaggle. This dataset contains 100 30-second samples for each of the following genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock.


## Multilayer Perceptron Neural Network (MLP)

An MLP has fully connected layers, which means all perceptrons are connected to all perceptrons in layers before or after them. This multi-layer perceptron network consists of an input layer, 3 hidden layers (ReLu), and an output layer (SoftMax). All layers are dense (fully connected)

My MLP Neural Network achieved a accuracy rate of 58.72% when we corrected for overfitting, and 62.53% when we did not. While this is below the rate an average human has for classifying genres (70%), it is still an interesting feat. It was right more than 50% of the time.

## Convolutional Neural Network (CNN)

CNNs are mainly used for image processing and image feature extraction, and are said to perform better than a multi-layer perceptron neural network. We will have three convolutional layers. The first two will have kernels of dimension 3 x 3. The last will be of size 2 x 2. The output layer will use softmax in order to classify the genre into one of our 10 classes.

My CNN achieved an accuracy rate of 71.74% when we corrected for overfitting, and 76.55% when we did not. I am really impressed with this. It has beaten the MLP, and is around and potentially even better the accuracy rate of a human (70%). I consider this a huge win.

I suspect the CNN beat the MLP because CNNs take in 2D images as input, while the spatial data is flattened with the MLP. Maybe by retaining that spatial information of the MFCCs, the CNN is more successful at identifying key audio features related to genre.

### Confusion Matrix

A confusion matrix is an effective way to show  how the model performed, and where its confidence lied when it comes to classifying certain groups.

<img width="653" alt="Screen Shot 2024-05-30 at 12 43 56 AM" src="https://github.com/NimunB/Music-Genre-Classification/assets/32827637/e9e6f626-dae2-4dd6-9560-6dcb397abfe5">


**Insights from Confusion Matrix**

- Pretty strong visual results. We see a clear and dark diagonal. Our most predicted label is always the correct one.
- Rock was the genre my model had the most confusion with. While it still predicted correctly most of the time (94 confidence score), it was most often confused with metal (39 score). This makes sense to me. Rock and metal are very similar genres and I would have trouble myself distinguishing between the two. It's interesting that the model corroborates that. Rock and Metal share the electric guitar, electric bass, and drums as their major instruments.
- My model seems to be the most confident with classifying classical music. It has the highest confidence score, and gets confused the least when classifying samples of this genre. My hypothesis as to why is because classical music is markedly older than the other genres the model was tested on, and hence its audio features including the MFCCs would be significantly different from the other more modern genres, and therefore easier to identify.
- HipHop was most often confused with Metal, Disco, and Reggae. This was a really interesting insight. Given that HipHop originated in the Bronx in New York in predominantly African American neighbourhoods, it makes sense that Reggae had an influence over the genre, and the effect is understood by the model as well. "Old School Hip Hop" was actually referred to as "Disco Rap" because so many hiphop beats were being made after re-sampling and synthesizing old disco records [10]. It's interesting that the model is picking up on these influences. I would love to further explore where the Metal confusion is coming from, because that it not an obvious genre I would confuse HipHop with. HipHop has a long history of being one of the outlets of people that were from disenfranchised communities. Metal is a genre associated with aggression and violent catharsis. Maybe (and this is just me making a conjecture) their audio featues are similar because both can involve catharsis and releasing negative emotions. I would be interested in exploring this more.


# What I learned

- How overfitting can be corrected using Dropout, Regularization, and stopping at an earlier epoch
- How genres like HipHop and Metal can be similar to other genres, and have that be represented in the confusuion matrix.


# Next Steps
- I would like to use more features in addition to MFCCs in the neural networks. Examples are Mel-spectrograms, Constant Q Chromas, and Tempograms.
- For comparison purposes, I think it would be interesting to try other classifiers like Recurrent Neural Networks (RNNs) and Support Vector Machines (SVMs).
