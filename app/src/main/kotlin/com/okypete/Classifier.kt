package com.okypete

import android.graphics.Bitmap
import com.okypete.data.models.RecognitionModel

/**
 * Generic interface for interacting with different recognition engines.
 */
interface Classifier {

    fun recognizeImage(bitmap: Bitmap): List<RecognitionModel>

    fun enableStatLogging(debug: Boolean)

    val statString: String

    fun close()
}

