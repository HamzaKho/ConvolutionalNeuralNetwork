This is a Convolutional Neural Network I have created using PyTorch.
It is trainedon the Fashion MNIST Dataset, and achieves an accuracy of 93%.

Structure –
- Initial Convoluton – the model starts with a Conv2d convolution layer that has 64 filters,
then a batch normalizaton. This processes the input.
- Residual Blocks – My model consists of 4 layers of BasicBlock modules. Each layer has varying
numbers of blocks and filters (64>128>256>512) respectvely. Each BasicBlock has 2
convolutonal layers and batch normalizatons. They also include a jump connecton.
- Adaptve pooling – After the residual blocks adaptve average pooling reduces the spatial
dimensions to 1x1.
- Fully connected Layer – There is a fully connected layer at the end that outputs the
probabilites of the input being in each class.

- Data preprocessing – The images are loaded and then normalised using the mean and standard
deviation of the dataset, this is to improve training stability and convergence speed. This also ensures
the model receives a standardized input, which is crucial for achieving effective learning.
- Learning Rate Optimizations – The learning rate is selected to be the most optimal, by gradually
increasing the learning rate while monitoring the loss. The learning rate where the loss decreases the
fastest is selected. This ensures efficient training.
Training Procedure – I used SGD with the optimal learning rate to train my model. The training
process involved minimising loss.
