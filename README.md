# Catdog
Cat vs Dog image recognizer NN

This project was for practicing a little bit on building own nns and also adopting pretrainted models. 

Therefore two approaches:
1) own simple nn with accuracy on validation set ~ 80.6% (catsanddogs.py)
2) model using pretrained RESNET18 with accuracy on validation set ~ 96% (catsanddogs_resnet_pretrained.py)

the folder doesn't contain the pictures.
you can download the pictures on: 
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

The own model contains just
 - 3 convolutional layer with batchnorm and maxpooling and followed by dropout layers.
 - 2 fully connected layers  

Special features:

- progress bars
- 8 pictures presented in testmode
- pretrained model flexible to change between other pretrained models

Still working on:
- auto-protocol function was adopted by another project of mine and needs some changes
- own nn very simple. will try to put in some residual layers on my own


   