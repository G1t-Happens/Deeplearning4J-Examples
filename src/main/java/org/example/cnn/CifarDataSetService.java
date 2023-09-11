package org.example.cnn;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.example.cnn.interfaces.DataSetService;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;


public class CifarDataSetService implements DataSetService {

    // Define some constants and instance variables
    // Define the input type for convolutional neural networks with dimensions 32x32x3
    private final InputType inputType = InputType.convolutional(32, 32, 3);

    // Define the number of training images
    private final int trainImagesNum = 512;

    // Define the number of test images
    private final int testImagesNum = 128;

    // Define the batch size for training
    private final int trainBatch = 16;

    // Define the batch size for testing
    private final int testBatch = 8;

    // Create instances of CifarDataSetIterator for training and testing
    private final CifarDataSetIterator trainIterator;
    private final CifarDataSetIterator testIterator;


    // Constructor initializes the iterators
    CifarDataSetService() {
        trainIterator = new CifarDataSetIterator(trainBatch, trainImagesNum, true);
        testIterator = new CifarDataSetIterator(testBatch, testImagesNum, false);
    }

    // Getter methods for accessing class properties
    public InputType getInputType() {
        return inputType;
    }

    public int getTrainImagesNum() {
        return trainImagesNum;
    }

    public int getTestImagesNum() {
        return testImagesNum;
    }

    public int getTrainBatch() {
        return trainBatch;
    }

    public int getTestBatch() {
        return testBatch;
    }

    public DataSetIterator getTrainIterator() {
        return trainIterator;
    }

    public DataSetIterator getTestIterator() {
        return testIterator;
    }

    // Implementing methods from the DataSetService interface
    @Override
    public DataSetIterator trainIterator() {
        return trainIterator;
    }

    @Override
    public DataSetIterator testIterator() {
        return testIterator;
    }

    @Override
    public InputType inputType() {
        return inputType;
    }

    @Override
    public List<String> labels() {
        return trainIterator.getLabels();
    }

}
