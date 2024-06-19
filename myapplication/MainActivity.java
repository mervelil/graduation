package com.example.myapplication;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    private final int IMAGE_PICK=100;
    Bitmap bitmap;
    Yolov5TFLiteDetector yolov5TFLiteDetector;
    Paint boxPaint =new Paint();
    Paint textPaint =new Paint();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        imageView =findViewById(R.id.imageView);
        yolov5TFLiteDetector =new Yolov5TFLiteDetector();
        yolov5TFLiteDetector.setModelFile("yolov5s-fp16.tflite");
        yolov5TFLiteDetector.initialModel(this);
        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setColor(Color.RED);

        textPaint.setTextSize(50);
        textPaint.setColor(Color.GREEN);
        textPaint.setStyle(Paint.Style.FILL);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
    public  void selectImage(View view){
        Intent intent =new Intent();
        intent.setAction(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent,IMAGE_PICK);

    }
    public  void predict(View view){


        ArrayList<Recognition> recognitions=yolov5TFLiteDetector.detect(bitmap)   ;
        Bitmap mutableBitmap=bitmap.copy(Bitmap.Config.ARGB_8888,true);
        Canvas canvas=new Canvas(mutableBitmap);
        for (Recognition recognition:recognitions) {
            if (recognition.getConfidence()>0.4) {
                RectF location =recognition.getLocation();

                canvas.drawRect(location,boxPaint);
                canvas.drawText(recognition.getLabelName()+":"+recognition.getConfidence(),location.left,location.top,textPaint);
            }

        }
        imageView.setImageBitmap(mutableBitmap);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==IMAGE_PICK&&data!=null) {
            Uri uri = data.getData();

            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }

    }
}