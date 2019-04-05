package com.tzutalin.dlibtest;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class Start_Screen extends Activity implements View.OnClickListener {

    Button button1,button2;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.start_screen);

        button1 = (Button) findViewById(R.id.button1);
        button1.setOnClickListener(this);

        button2 = (Button) findViewById(R.id.button2);
        button2.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.button1:
                Intent intent1 = new Intent(this, Show_Features.class);
                startActivity(intent1);
                break;

            case R.id.button2:
                Intent intent2 = new Intent(this, Processing.class);
                startActivity(intent2);
                break;

            default:
                break;
        }
    }
}
