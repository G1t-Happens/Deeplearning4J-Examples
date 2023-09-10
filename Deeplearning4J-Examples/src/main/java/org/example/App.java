package org.example;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;


public class App {


    private static final int CLASSES_COUNT = 3;
    private static final int FEATURES_COUNT = 4;

    public static void main(String[] args) throws IOException, InterruptedException {

        DataSet allDataSetData;
        //Reading file(irisdataset.txt) with 0 skips and delimiter(,)
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("irisdataset.txt").getFile()));

            //Small data set of 150 records -> read all the data into memory at once with a call of iterator.next()
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            allDataSetData = iterator.next();
        }

        //shuffle the dataset to get rid of ordering of classes
        //constant seed of 45 because we always want the same shuffling results
        allDataSetData.shuffle(45);

        //Normalizing(normal distribution) our data -> https://www.baeldung.com/cs/normalize-table-features
        DataNormalization normalize = new NormalizerStandardize();
        normalize.fit(allDataSetData); //gathering tatistics about the data like mean and standard deviation
        normalize.transform(allDataSetData); //data is adjusted so that it has a mean of 0 and a standard deviation of 1

        //Splitting data set into training and test data
        SplitTestAndTrain testAndTrainSet = allDataSetData.splitTestAndTrain(0.70);
        DataSet trainingDataSet = testAndTrainSet.getTrain(); //Getting the trainingDataSet(150*0.7=105)
        DataSet testingDataSet = testAndTrainSet.getTest(); //Getting the testingDataSet(150*0.3=45)

        //Network Configuration(Configuration Builder)
        MultiLayerConfiguration configurationOne = new NeuralNetConfiguration.Builder()
                //optimization iterations
                .iterations(2000)
                //determine the output of a neuron given its input(hyperbolic tangent)
                .activation(Activation.TANH)
                //initial weights -> helps with better convergence during training
                .weightInit(WeightInit.XAVIER)
                //learning rate controls the step size during gradient descent
                .learningRate(0.1)
                //L2 regularization helps prevent overfitting by adding a penalty term
                .regularization(true).l2(0.0001)
                //initializes a list to define the layers of the neural network
                .list()
                //adds the first hidden layer to the network. It's a dense (fully connected) layer with FEATURES_COUNT input neurons and 3 output neurons.
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                //adds a second hidden layer with 3 input neurons and 3 output neurons.
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                //This adds the output layer, which is configured for multi-class classification using negative log likelihood
                //as the loss function and softmax activation. It has 3 input neurons and CLASSES_COUNT output neurons,
                //which correspond to the classes in the dataset.
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false) //backpropagation should be used for training the network
                .build();

        MultiLayerConfiguration configurationTwo = new NeuralNetConfiguration.Builder()
                .iterations(2000)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.01)
                .updater(new Adam())
                .regularization(true).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(FEATURES_COUNT)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(32)
                        .nOut(CLASSES_COUNT)
                        .build())
                .setInputType(InputType.convolutionalFlat(FEATURES_COUNT, 1, 1))
                .backprop(true).pretrain(false)
                .build();

        // Define an array or list of configurations
        MultiLayerConfiguration[] configurations = new MultiLayerConfiguration[]{
                configurationOne,
                configurationTwo,
                // Add more configurations as needed
        };

        int configIndex = 1;
        for (MultiLayerConfiguration configuration : configurations) {

            //Creating and Training a Network
            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            //initialize mode
            model.init();
            //trains the neural network using the provided training dataset
            model.fit(trainingDataSet);

            //trained model is used to make predictions on a separate testing dataset
            INDArray output = model.output(testingDataSet.getFeatures());

            //evaluate the performance of the model, specified by the number of classes
            Evaluation eval = new Evaluation(CLASSES_COUNT);
            //evaluates the model's predictions by comparing them to the actual labels of the testing dataset.
            //computes various metrics such as accuracy, precision, recall, and F1-score.
            eval.eval(testingDataSet.getLabels(), output);

            //Yeahh results :DD
            System.out.print("\nConfiguration " + configIndex + " Results:");
            System.out.println(eval.stats());
            System.out.println();
            System.out.println();
            configIndex++;
        }
    }

}
