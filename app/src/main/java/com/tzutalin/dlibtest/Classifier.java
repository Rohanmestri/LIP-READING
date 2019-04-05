package com.tzutalin.dlibtest;

public interface Classifier {
    String name();

    float[] recognize(final float[] pixels);
}