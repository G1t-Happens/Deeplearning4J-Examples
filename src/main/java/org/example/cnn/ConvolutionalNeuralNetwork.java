package org.example.cnn;

import org.deeplearning4j.eval.Evaluation;


public class ConvolutionalNeuralNetwork {

    //build and train a convolutional neural network(https://de.wikipedia.org/wiki/Convolutional_Neural_Network)
    public static void main(String... args) {
        ConvolutionalNeuralNetworkModel convolutionalNeuralNetworkModel = new ConvolutionalNeuralNetworkModel(new CifarDataSetService());
        convolutionalNeuralNetworkModel.trainModel();
        Evaluation evaluation = convolutionalNeuralNetworkModel.evaluateTrainedModel();
        System.out.println("Lets Go :D");
        System.out.println(evaluation.stats());
    }
}
