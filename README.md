# Text Classification with Convolutional Neural Network

## 1. required environment
environment |  version 
----------- | ---------
numpy       |  1.13.3
nltk        |  3.2.5
pandas      |  0.22.0
gensim      |  3.4.0
jieba       |  0.39
sklearn     |  0.19.1
tensorflow  |  1.6.0



## 2.  description

The architecture of text cnn can be depicted as the following figure, it has been referred in the paper that single filter can extract a specific feature after convolution operation, applying the pooling to the results, a sentence can be transformed into a vector, if we use max pooling ,we will get a scalar, in order to improve the accuracy and complexity of features, it can be considered to take some filters with different size, while it can extract multi-scale features, as the result, a sentence will be decoded into a vector which has the same size as the amount of filters. 

The project use four filters with different size 3, 4, 5, 6 and each filter'num is 80, you can also set your own filters as it can performs excellent on the datasets.  after convolutional layer, it should get a matrix which has shape [batch-size, 4 * 80],  in the fully connected layer, the matirx will become [batch-size, hidden-size], which the hidden_size denotes the number of neurons in the hidden layer. In the final stage, the sentences can be encoded into a matrix with shape [batch-size, classes-num] connected to a softmax layer.

![text cnn](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Convolutional-Neural-Network-Architecture-for-Sentence-Classification.png)


## Reference
1. [Convolutional Neural Network for Sentence Classification. [Yoon Kim]. 2014](https://arxiv.org/pdf/1408.5882.pdf)

