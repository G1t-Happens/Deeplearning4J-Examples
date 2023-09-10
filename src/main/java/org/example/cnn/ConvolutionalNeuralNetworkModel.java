package org.example.cnn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.example.cnn.interfaces.DataSetService;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ConvolutionalNeuralNetworkModel {


    private final DataSetService dataSetService;

    private final MultiLayerNetwork multiLayerNetwork;


    public ConvolutionalNeuralNetworkModel(DataSetService dataSetService) {

        this.dataSetService = dataSetService;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1500)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .regularization(true)
                .updater(Updater.ADAM)
                .list()
                .layer(0, convolutionLayer5x5())
                .layer(1, subsamplingLayerPooling2x2Stride2())
                .layer(2, convolutionLayer3x3Stride1Padding2())
                .layer(3, subsamplingLayerPooling2x2Stride1())
                .layer(4, convolutionLayer3x3Stride1Padding1())
                .layer(5, subsamplingLayerPooling2x2Stride1())
                .layer(6, outputLayerDense())
                .pretrain(false)
                .backprop(true)
                .setInputType(dataSetService.inputType())
                .build();
        multiLayerNetwork = new MultiLayerNetwork(configuration);
    }

    private ConvolutionLayer convolutionLayer5x5() {
        return new ConvolutionLayer.Builder(5, 5)
                .nIn(3)
                .nOut(16)
                .stride(1, 1)
                .padding(1, 1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build();
    }

    private SubsamplingLayer subsamplingLayerPooling2x2Stride2() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();
    }

    private ConvolutionLayer convolutionLayer3x3Stride1Padding2() {
        return new ConvolutionLayer.Builder(3, 3)
                .nOut(32)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build();
    }

    private SubsamplingLayer subsamplingLayerPooling2x2Stride1() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build();
    }

    private ConvolutionLayer convolutionLayer3x3Stride1Padding1() {
        return new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .padding(1, 1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .activation(Activation.RELU)
                .build();
    }

    private OutputLayer outputLayerDense() {
        return new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .nOut(dataSetService.labels().size() - 1)
                .build();
    }

    //TODO: MultiLayerConfiguration, train, evaluate, ConvolutionLayer, SubsamplingLayer, OutputLayer
}
