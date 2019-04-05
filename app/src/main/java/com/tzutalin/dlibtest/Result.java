package com.tzutalin.dlibtest;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import java.util.Arrays;
import java.util.List;

public class Result extends Activity {

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_display_message);

        // Get the Intent that started this activity and extract the string
        b = this.getIntent().getExtras();
        float[] array = b.getFloatArray("1");

        List<String> words = Arrays.asList("Abuse", "Black", "Exactly", "Missing");

        // Capture the layout's TextView and set the string as its text
        TextView textView = findViewById(R.id.textView);
        for (int i = 0; i < array.length; i++)
            textView.append(words.get(i) + ": " +toString().valueOf(array[i]) + "\n" + "\n");
    }
}
