package com.okypete.data.models

import android.graphics.RectF

/**
 * @author lusinabrian on 17/09/17.
 * @Notes An immutable result returned by a Classifier describing what was recognized.
 *
 * @param id A unique identifier for what has been recognized. Specific to the class, not the
 * instance ofvthe object.
 * @param title  Display name for the recognition.
 * @param confidence A sortable score for how good the recognition is relative to others.
 * Higher should be better.
 * @param location Optional location within the source image for the location of the recognized object.
*/
data class RecognitionModel(
        val id: String?,
        val title: String?,
        val confidence: Float?,
        private var location: RectF?
){

    fun getLocation(): RectF {
        return RectF(location)
    }

    override fun toString(): String {
        var resultString = ""
        if (id != null) {
            resultString += "[$id] "
        }

        if (title != null) {
            resultString += title + " "
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f)
        }

        if (location != null) {
            resultString += location!!.toString() + " "
        }

        return resultString.trim { it <= ' ' }
    }
}
