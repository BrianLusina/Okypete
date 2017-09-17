package com.okypete;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;
import android.util.Log;

import com.okypete.data.models.RecognitionModel;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * @author lusinabrian on 17/09/17.
 * @Notes A classifier specialized to label images using TensorFlow.
 */

public class TFImageClassifier implements Classifier {

  private static final String TAG = "TFImageClassifier";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;

  // Config values.
  private String inputName;
  private String outputName;
  private int inputSize;
  private int imageMean;
  private float imageStd;

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<>();
  private int[] intValues;
  private float[] floatValues;
  private float[] outputs;
  private String[] outputNames;

  private TensorFlowInferenceInterface inferenceInterface;

  private TFImageClassifier() {
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
   * @param imageMean The assumed mean of the image values.
   * @param imageStd The assumed std of the image values.
   * @param inputName The label of the image input node.
   * @param outputName The label of the output node.
   */
  public static Classifier create(
      AssetManager assetManager,
      String modelFilename,
      String labelFilename,
      int inputSize,
      int imageMean,
      float imageStd,
      String inputName,
      String outputName)
      throws IOException {
    TFImageClassifier c = new TFImageClassifier();
    c.inputName = inputName;
    c.outputName = outputName;

    // Read the label names into memory.
    // TODO: make this handle non-assets.
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    Log.i(TAG, "Reading labels from: " + actualFilename);
    BufferedReader br;
    br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
    String line;
    while ((line = br.readLine()) != null) {
      c.labels.add(line);
    }
    br.close();

    c.inferenceInterface = new TensorFlowInferenceInterface();
    if (c.inferenceInterface.initializeTensorFlow(assetManager, modelFilename) != 0) {
      throw new RuntimeException("TF initialization failed");
    }
    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
    int numClasses =
        (int) c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1);
    Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

    // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
    // the placeholder node for input in the graphdef typically used does not specify a shape, so it
    // must be passed in as a parameter.
    c.inputSize = inputSize;
    c.imageMean = imageMean;
    c.imageStd = imageStd;

    // Pre-allocate buffers.
    c.outputNames = new String[]{outputName};
    c.intValues = new int[inputSize * inputSize];
    c.floatValues = new float[inputSize * inputSize * 3];
    c.outputs = new float[numClasses];

    return c;
  }

  @Override
  public List<RecognitionModel> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
    }
    Trace.endSection();

    // Copy the input data into TensorFlow.
    Trace.beginSection("fillNodeFloat");
    inferenceInterface.fillNodeFloat(
        inputName, new int[]{1, inputSize, inputSize, 3}, floatValues);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("runInference");
    inferenceInterface.runInference(outputNames);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("readNodeFloat");
    inferenceInterface.readNodeFloat(outputName, outputs);
    Trace.endSection();

    // Find the best classifications.
    PriorityQueue<RecognitionModel> pq =
        new PriorityQueue<>(
            3,
            new Comparator<RecognitionModel>() {
              @Override
              public int compare(RecognitionModel lhs, RecognitionModel rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(
            new RecognitionModel(
                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
      }
    }
    final ArrayList<RecognitionModel> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(boolean debug) {
    inferenceInterface.enableStatLogging(debug);
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
