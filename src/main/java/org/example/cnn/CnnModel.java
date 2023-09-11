package org.example.cnn;

import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class CnnModel {

    protected static final Logger log = LoggerFactory.getLogger(CnnModel.class);

    private final int numLabels = CifarLoader.NUM_LABELS;
    private final int batchSize = 96;
    private final int height = 32;
    private final int width = 32;
    private final int channels = 3;
    private final long seed = 123L;
    private final int epochs = 5;

    public MultiLayerNetwork getMultiLayerNetworkModel() {
        log.info("Start building MultiLayerConfiguration");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(layer_0())
                .layer(layer_1())
                .layer(layer_2())
                .layer(layer_3())
                .layer(layer_4())
                .layer(layer_5())
                .layer(layer_6())
                .layer(layer_7())
                .layer(layer_8())
                .layer(layer_9())
                .layer(layer_10())
                .layer(layer_11())
                .layer(layer_12())
                .layer(layer_13())
                .layer(layer_14())
                .layer(layer_15())
                .layer(layer_16())
                .layer(layer_17())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        log.info("Done building MultiLayerConfiguration");
        log.info("INIT - MultiLayerConfiguration");
        multiLayerNetwork.init();
        return multiLayerNetwork;
    }

    //Layer 0
    private ConvolutionLayer layer_0() {
        return new ConvolutionLayer.Builder()
                .kernelSize(3, 3)
                .stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nIn(channels)
                .nOut(32)
                .build();
    }

    //Layer 1
    private BatchNormalization layer_1() {
        return new BatchNormalization();
    }

    //Layer 2
    private SubsamplingLayer layer_2() {
        return new SubsamplingLayer.Builder()
                .kernelSize(2, 2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build();
    }

    //Layer 3
    private ConvolutionLayer layer_3() {
        return new ConvolutionLayer.Builder()
                .kernelSize(1, 1).stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(16)
                .build();
    }

    //Layer 4
    private BatchNormalization layer_4() {
        return new BatchNormalization();
    }

    //Layer 5
    private ConvolutionLayer layer_5() {
        return new ConvolutionLayer.Builder()
                .kernelSize(3, 3)
                .stride(1, 1).padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(64)
                .build();
    }

    //Layer 6
    private BatchNormalization layer_6() {
        return new BatchNormalization();
    }

    //Layer 7
    private SubsamplingLayer layer_7() {
        return new SubsamplingLayer.Builder()
                .kernelSize(2, 2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build();
    }

    //Layer 8
    private ConvolutionLayer layer_8() {
        return new ConvolutionLayer.Builder()
                .kernelSize(1, 1)
                .stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(32)
                .build();
    }

    //Layer 9
    private BatchNormalization layer_9() {
        return new BatchNormalization();
    }

    //Layer 10
    private ConvolutionLayer layer_10() {
        return new ConvolutionLayer.Builder()
                .kernelSize(3, 3)
                .stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(128).build();
    }

    //Layer 11
    private BatchNormalization layer_11() {
        return new BatchNormalization();
    }

    //Layer 12
    private ConvolutionLayer layer_12() {
        return new ConvolutionLayer.Builder()
                .kernelSize(1, 1)
                .stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(64)
                .build();
    }

    //Layer 13
    private BatchNormalization layer_13() {
        return new BatchNormalization();
    }

    //Layer 14
    private ConvolutionLayer layer_14() {
        return new ConvolutionLayer.Builder()
                .kernelSize(1, 1)
                .stride(1, 1)
                .padding(1, 1)
                .activation(Activation.LEAKYRELU)
                .nOut(numLabels)
                .build();
    }

    //Layer 15
    private BatchNormalization layer_15() {
        return new BatchNormalization();
    }

    //Layer 16
    private SubsamplingLayer layer_16() {
        return new SubsamplingLayer.Builder()
                .kernelSize(2, 2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.AVG)
                .build();
    }

    //Layer 17
    private OutputLayer layer_17() {
        return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .dropOut(0.8)
                .activation(Activation.SOFTMAX)
                .build();
    }




    public int getNumLabels() {
        return numLabels;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public int getChannels() {
        return channels;
    }

    public long getSeed() {
        return seed;
    }

    public int getEpochs() {
        return epochs;
    }
}
