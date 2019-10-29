# TensorFlow2.0-Tutorial
A TensorFlow2.0 tutorial for teaching purpose. (Course ended at Sep. 2019)

## Dataset
Note that the folders for dataset and checkpoints are not uploaded in GitHub, so things may broke somewhere.

## Temptatvie Schedule
1. Simple network
	* TensorFlow installation
	* Tensor and eager execution
	* Simple image classification example (ex. MINST)
	* Data preprocessing
	* Build model using tf.keras.Sequential
	* Inspect model and plot model graph
	* Keras callback
	* Save and load model
	* Add convolutional layers to model
	* Simple regression example (Auto MPG) (optional)
2. Keras functional API
    * Simple ResNet model
    * Building residual block
    * Complex graph topologies (ex. [Image Colorization](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf))
    * Model with shared layers
    * Model with multiple inputs and ouputs
    * Nice coding practices
3. Data input pipeline using TensorFlow Dataset
    * tf.data input pipeline (ex. AOI image classification)
    * Train/validation split
    * Data augmentation
    * Write loss and metrics to csv file
    * Train model on cloud virtual machine
    * Short guide on GPU choice
4. Transfer learning
    * Transfer learning with pretrained CNN
5. Generative models
    * DCGAN (Generating anime faces using Getchu dataset)
    * Constructing data input pipeline from TFRecord
    * Custom training loop with tf.GradientTape()

## Contact
Please file an issue if there's any error, or suggest better coding practice. Thanks!