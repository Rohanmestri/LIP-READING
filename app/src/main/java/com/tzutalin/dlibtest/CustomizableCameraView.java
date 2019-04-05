package com.tzutalin.dlibtest;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;

import org.opencv.android.JavaCameraView;

public class CustomizableCameraView extends JavaCameraView {

    public CustomizableCameraView(Context context, int id) {
        super(context, id);
    }

    public void setPreviewFPS(double min, double max){
        Camera.Parameters params = mCamera.getParameters();
        params.setPreviewFpsRange((int)(min*1000), (int)(max*1000));
        mCamera.setParameters(params);
    }
}
