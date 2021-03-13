# perceptron
simple perceptron to recognize 28x28 sized images. Digits between 0-9. 
### Describe the algorithm:

*Inputs*: 28x28 sized image with handwritten digits

![](sample-digit/0/10.png) ![](sample-digit/1/1004.png) ![](sample-digit/2/1.png) ![](sample-digit/3/1020.png) ![](sample-digit/4/1010.png) ![](sample-digit/5/102.png) ![](sample-digit/6/11.png) ![](sample-digit/7/0.png) ![](sample-digit/8/61.png) ![](sample-digit/9/7.png) 

*Feature*: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit
         of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias).
         
*Label*: Actual label of the train feature.

weight_i2h: current weights with shape (in_dim x hidden_dim) from input (feature vector) to hidden layer.

weight_h2o: current weights with shape (hidden_dim x out_dim) from hidden layer to output (digit number 0-9).

