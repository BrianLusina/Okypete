package com.okypete

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.flurgle.camerakit.CameraListener
import com.flurgle.camerakit.CameraView
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {

        private val INPUT_SIZE = 224
        private val IMAGE_MEAN = 117
        private val IMAGE_STD = 1f
        private val INPUT_NAME = "input"
        private val OUTPUT_NAME = "output"

        private val MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb"
        private val LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt"
    }

    private var classifier: Classifier? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var textViewResult: TextView? = null
    private var btnDetectObject: Button? = null
    private var btnToggleCamera: Button? = null
    private var imageViewResult: ImageView? = null
    private var cameraView: CameraView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        cameraView = findViewById<View>(R.id.cameraView) as CameraView
        imageViewResult = findViewById<View>(R.id.imageViewResult) as ImageView
        textViewResult = findViewById<View>(R.id.textViewResult) as TextView
        textViewResult!!.movementMethod = ScrollingMovementMethod()

        btnToggleCamera = findViewById<View>(R.id.btnToggleCamera) as Button
        btnDetectObject = findViewById<View>(R.id.btnDetectObject) as Button

        cameraView!!.setCameraListener(object : CameraListener() {
            override fun onPictureTaken(picture: ByteArray?) {
                super.onPictureTaken(picture)

                var bitmap = BitmapFactory.decodeByteArray(picture, 0, picture!!.size)

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)

                imageViewResult!!.setImageBitmap(bitmap)

                val results = classifier!!.recognizeImage(bitmap)

                textViewResult!!.text = results.toString()
            }
        })

        btnToggleCamera!!.setOnClickListener { cameraView!!.toggleFacing() }

        btnDetectObject!!.setOnClickListener { cameraView!!.captureImage() }

        initTensorFlowAndLoadModel()
    }

    override fun onResume() {
        super.onResume()
        cameraView!!.start()
    }

    override fun onPause() {
        cameraView!!.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.execute { classifier!!.close() }
    }

    private fun initTensorFlowAndLoadModel() {
        executor.execute {
            try {
                classifier = TensorFlowImageClassifier.create(
                        assets,
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME)
                makeButtonVisible()
            } catch (e: Exception) {
                throw RuntimeException("Error initializing TensorFlow!", e)
            }
        }
    }

    private fun makeButtonVisible() {
        runOnUiThread { btnDetectObject!!.visibility = View.VISIBLE }
    }
}
