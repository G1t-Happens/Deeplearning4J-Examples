package org.example.cnn;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ConvolutionalNeuralNetwork {


    protected static final Logger log = LoggerFactory.getLogger(ConvolutionalNeuralNetwork.class);


    public static void main(String[] args) {

        //Create an instance of the CnnModel class
        log.info("Create an instance of the CnnModel class");
        CnnModel cnnModel = new CnnModel();

        // Create data iterators for training and testing using the CIFAR-10 dataset
        log.info("Create data iterators for training and testing using the CIFAR-10 dataset");
        Cifar10DataSetIterator cifarTrainIterator = new Cifar10DataSetIterator(cnnModel.getBatchSize(),
                new int[]{cnnModel.getHeight(), cnnModel.getWidth()}, DataSetType.TRAIN, null, cnnModel.getSeed());

        Cifar10DataSetIterator cifarTestIterator = new Cifar10DataSetIterator(cnnModel.getBatchSize(),
                new int[]{cnnModel.getHeight(), cnnModel.getWidth()}, DataSetType.TEST, null, cnnModel.getSeed());

        // Get a pre-defined multi-layer neural network model
        log.info("Get a pre-defined multi-layer neural network model");
        MultiLayerNetwork multiLayerNetwork = cnnModel.getMultiLayerNetworkModel();

        // Set listeners to track and log model training progress
        log.info("Set listeners to track and log model training progress -> IterationLis..(50) & EvaluationL..(1)");
        multiLayerNetwork.setListeners(new ScoreIterationListener(50),
                new EvaluativeListener(cifarTestIterator, 1, InvocationType.EPOCH_END));

        // Train the neural network using the training data
        log.info("Start training the neural network using the training data with: {} epochs", cnnModel.getEpochs());
        multiLayerNetwork.fit(cifarTrainIterator, cnnModel.getEpochs());
    }

}

