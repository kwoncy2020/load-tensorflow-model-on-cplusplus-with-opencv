# load-tensorflow-model-on-cplusplus-with-opencv   
load tensorflow model saved from the way of saved_model in python to c++ with opencv    

for someone who look for a way to bring the model made by python-tensorflow to c++. Hopefully, this may help someone who spend tons of time for search the way of use tensorflow-c-api with 2d convolution model under the poor infomations on online. I tested this only on windows 10, c++14, visual studio 2022 in my mfc project.
if there something goes wrong when you try, but still you can get some insights through my code. cheers!
***
this will use tensorflow-c-api and opencv in c++.   

before use this file, you need to install tensorflow, tensorflow-c-api and opencv in your system and you also need to configure your project by add dll, lib, etc.   
after finishing settings, include files(LoadDeepModel_tf.cpp, LoadDeepModel_tf.h) to your project.   

***
***
***

\*notice!    
\*you cannot use model saved by 'h5'. if you have a model like this('xxxxx.h5'), then you need to save your model through the way of saved_model.   
\*when you have a model saved by the way of saved_model, the folder might seem like below.   
* saved_model  
  * mymodel   
     * assets   
     * variables   
     * keras_metadata.pb   
     * saved_model.pb   


if you have a proper model, you need to extract some informations about your model through saved_model_cli which is kind of command you can run under the command prompt.   

four arguments are required for load saved_model saved by keras api  (the path of saved_model, the input name of signature_def, the output name of signature_def and tag name).   

you can find it from using saved_model_cli. you need to find the way of using saved_model_cli.     

ex) C:\saved_model_cli show --dir {saved_model_folder\my_model_forder}  
> ==> tag-sets: 'serve' ( you need to check if it 'serve' or not. if not, using that name)   

ex) C:\saved_model_cli show --dir {saved_model_folder\my_model_forder} --tag_set serve  
> ==> SignatureDef key: "__saved_model_init_op"
                   SignatureDef key: "serving_default"   

ex) C:\saved_model_cli show --dir {saved_model_folder\my_model_forder} --tag_set serve --signature_def serving_default 

> The given SavedModel SignatureDef contains the following output(s):   
      inputs['input_1'] tensor_info:   
          dtype: DT_FLOAT   
  				shape: (-1, 240, 320, 1)   
  				name: serving_default_input_1:0   
  		The given SavedModel SignatureDef contains the following output(s):   
  			outputs['conv2d_10'] tenfor_info:   
  				dtype: DT_FLOAT   
  				shape: (-1, 240, 320, 1)   
  				name: StatefulPartitionedCall:0   
  		Method name is : tensorflow/serving/predict   
 
you can see like this. in this case, next four arguments are requred.    
>  mymodel-forder-path  ( * not the model saved by the way of h5. the folder should contain the model saved by the way of saved_model )    
  serving_default_input_1  ( you can see this on the name of inputs['input 1'] tensor_info )   
  StatefulPartitionedCall  ( you can see this on the name of outputs['conv2d 10'] tensor_info )   
  serve ( you can see this on tag-sets)   

***
***
***

now you can include files(LoadDeepModel.cpp, LoadDeepModel.h) to your main file.    

\* notice! when you using this LoadDeepModel class, there might be stack-overflow on the stack. Because this handle images, you need to expand stack size on your project property settings.    
\* It's obvious to make the image's size same with the model you loaded. please make sure your Image and model's shape.

you can make instance with constructor which requires four args that you already checked.   

```cpp
// in my case this will be LoadDeepModel LoadDeepModel(R"(C:\saved_model\mymodel)", "serving_default_input_1", "StatefulPartitionedCall", "serve");   
LoadDeepModel LoadDeepModel(SAVED_MODEL_PATH, MODEL_INPUT_SIGNATURE, MODEL_OUTPUT_SIGNATURE, MODEL_TAG_NAME);

//you can check weather it loaded or not by below way. if its successful, it will return TF_OK   
if (TF_GetCode(LoadDeepModel.status) != TF_OK) return -1;   

// now you can prepare an image and put it into .predict() method.
cv::Mat mat;   
mat = cv::imread(R"(C:\picture\0.png)", cv::IMREAD_COLOR);  // the loaded image has already same shape with the model in my case. you might need to resize your loaded image.
 
cv::Mat mat_gray;
cv::cvtColor(mat, mat_gray, cv::COLOR_BGR2GRAY);
 
cv::Mat matForPredict;
mat_gray.convertTo(matForPredict, CV_32FC1);
 
cv::Mat predicted = LoadDeepModel.predict(&matForPredict, 0.5);  // 0.5 means threshold. the predicted type is 8UC1.
 
cv::Mat out = LoadDeepModel.getResizedImageWithMask(mat, predicted, 2);  // 2 means scale-factor. you can use 2,4,8 scale.
 
cv::imshow("pred", out);    // if its successful, you can see a red area of segmented by model.   
cv::waitKey(0);   
cv::destroyAllWindows();   
```

***
 ![1](https://user-images.githubusercontent.com/96859911/170189820-5698076d-1a07-44cc-a207-c41b2bc8532a.png)

 
 
