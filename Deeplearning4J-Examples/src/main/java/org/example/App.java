package org.example;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

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
    }

}
