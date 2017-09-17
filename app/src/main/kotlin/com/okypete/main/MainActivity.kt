package com.okypete.main

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.text.method.ScrollingMovementMethod
import android.view.View
import com.flurgle.camerakit.CameraListener
import com.okypete.*
import kotlinx.android.synthetic.main.activity_main.*
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity(), MainView {

    private lateinit var classifier: Classifier
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        textViewResult.movementMethod = ScrollingMovementMethod()

        cameraView.setCameraListener(object : CameraListener() {
            override fun onPictureTaken(picture: ByteArray?) {
                super.onPictureTaken(picture)

                var bitmap = BitmapFactory.decodeByteArray(picture, 0, picture!!.size)

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)

                imageViewResult.setImageBitmap(bitmap)

                val results = classifier.recognizeImage(bitmap)

                textViewResult.text = results.toString()
            }
        })

        btnToggleCamera.setOnClickListener { cameraView.toggleFacing() }

        btnDetectObject.setOnClickListener { cameraView.captureImage() }

        initTensorFlowAndLoadModel()
    }

    override fun onResume() {
        super.onResume()
        cameraView.start()
    }

    override fun onPause() {
        cameraView.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute { classifier.close() }
    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute {
            try {
                classifier = TFImageClassifier.create(assets, MODEL_FILE, LABEL_FILE,
                        INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME)
                runOnUiThread {
                    btnDetectObject.visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        }
    }
}
