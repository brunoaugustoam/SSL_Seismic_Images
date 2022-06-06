# SSL_Seismic_Images

Codes of the paper "Self-Supervised Learning for Seismic Image Segmentation from Few-Labeled Samples", submitted to the special stream of IEEE Geoscience and Remote Sensing letters at Sibgrapi 2022.

Google Colab Demo can be found [[here]](https://colab.research.google.com/drive/1CS5itL_a_Ojwn0Pbuy_c4YDh9u20U57_?authuser=2).

Authored by:
* Bruno A. A. Monteiro - Department of Computer Science, UFMG, Belo Horizonte, Brazil
* Hugo Oliveira - Institute of Mathematics and Statistics, USP, São Paulo, Brazil
* Jefersson A. dos Santos - Department of Computer Science, UFMG, Belo Horizonte, Brazil

# Method


## Datasets
 * Netherlands F3 Interpretation Dataset [[link]](https://zenodo.org/record/1471548\#.Yf0Y3-rMKrx). 
 * Parihaka Seismic Data [[link]](https://public.3.basecamp.com/p/JyT276MM7krjYrMoLqLQ6xST). 

## Pretext Tasks
 * Rotation: the original image is randomly rotated into one of the five possible angles (-8,-4, 0, 4, 8 degrees). Then it is cropped, so no empty values are given as input. The network must then identify which of the rotations was applied. 
 * Jigsaw: the original image is cropped into 9 regular-sized tiles with a small random gap between them. To avoid an overly complicated task, the possible permutations are limited to 1000, being 500 randomly selected, and for each of them, its pair with the biggest hamming distance is selected. The model is requested to determine the original position for each one of the permuted tiles. 

When training the model to solve the pretext tasks, the entire train set is used to train, and the validation set is used for evaluation. 

## Fine-Tune

We randomly selected 1, 5, 10, and 20 labeled seismic sections from the train set, and used only them for fine-tuning. Later on, testing into the entire test set of each dataset.

## Backbone

* ResNet-50 provided by PyTorch [[link]](https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html).

For the classification tasks, the output of the backbone is connected to an average pooling layer followed by the fully-connected classifier. For the segmentation task, we utilize a bilinear interpolation to retrieve the original input size, concatenating the first and the last activations. Then, two final convolutional layers are used to segment the dataset classes.  

Dropout was applied after each one of the four layer sets. All training images were augmented using random crop, half-chance horizontal flip, and Gaussian noise addition. We ran a grid search to find the optimal setups for both pretext tasks, varying the learning rates, the optimizers, and schedulers. The best models were then used for fine-tuning. In this stage, the employed grid search varies only the learning rates, using Adam optimizer and StepLR. The best setup was then selected for 5-fold cross-validation, being the initial learning rate of $1 \times 10^{-4}$ for the backbone and $1 \times 10^{-3}$ for the segmenter. For the baseline, the best setup was employing 10x bigger learning rates. At every stage, we use a weight decay of $1e^{-4}$. To validate our results, we calculated the confidence interval assuming a two-tailed paired t-student distribution.

## Requires 
- matplotlib
- numpy
- scikit-image
- scikit-learn
- segyio
- torch
- torchinfo
- torchvision
