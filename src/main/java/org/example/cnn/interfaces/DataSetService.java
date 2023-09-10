package org.example.cnn.interfaces;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;


public interface DataSetService {


    // Return a DataSetIterator for training data
    DataSetIterator trainIterator();

    // Return a DataSetIterator for testing data
    DataSetIterator testIterator();

    // Return the InputType, which describes the data format expected by a neural network
    InputType inputType();

    // Return a list of labels associated with the dataset
    List<String> labels();
}
