package com.okypete.main

import android.os.Bundle

/**
 * @author lusinabrian on 17/09/17.
 * @Notes main presenter
 */
interface MainPresenter<V : MainView> {
    /**
     * on view created, we create from saved instance if available*/
    fun onViewCreated(savedInstanceState : Bundle?)
}
