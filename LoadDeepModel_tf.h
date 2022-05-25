#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
// four arguments are required for load saved_model saved by keras api  (the path of saved_model, the input name of signature_def, the output name of signature_def and tag name)
// you can find it from using saved_model_cli. you need to find the way of install tensorflow and of using saved_model_cli.  
// ex) C:\saved_model_cli show --dir {saved_model_folder\my_model_forder}  ==> return tag-sets: 'serve' ( you need to check if it 'serve' or not. if not, using that name)
// ex) C:\saved_model_cli show --dir {saved_model_folder\my_model_forder} --tag_set serve  ==> return SignatureDef key: "__saved_model_init_op"
//                                                                                                    SignatureDef key: "serving_default"
// ex) C:\saved_model_cli shiw --dir {saved_model_folder\my_model_forder} --tag_set serve --signature_def serving_default ==> return some info. see below.
//		The given SavedModel SignatureDef contains the following output(s):
//			inputs['input_1'] tensor_info:
//				dtype: DT_FLOAT
//				shape: (-1, 240, 320, 1)
//				name: serving_default_input_1:0
//		The given SavedModel SignatureDef contains the following output(s):
//			outputs['conv2d_10'] tenfor_info:
//				dtype: DT_FLOAT
//				shape: (-1, 240, 320, 1)
//				name: StatefulPartitionedCall:0
//		Method name is : tensorflow/serving/predict
// 
// you can see like this. in this case, next four arguments are requred.
//	my_model_forder  ( * not the model saved by the way of h5. the folder should contain the model saved by the way of saved_model ) 
//  serving_default_input_1
//	StatefulPartitionedCall
//  serve
//


class LoadDeepModel
{
public:
	
	LoadDeepModel(std::string str_model_path, std::string str_model_input_signature, std::string str_model_output_signature, std::string str_model_tag_name);
	void setModel(std::string model_path, std::string str_model_input_signature, std::string str_model_output_signature, std::string str_model_tag_name);
	bool loadModel();
	bool isModelLoaded();
	cv::Mat predict(cv::Mat* mat, float threshold);
	cv::Mat getResizedImageWithMask(cv::Mat palette, cv::Mat mask, int modifyRate);

	TF_Status* status = TF_NewStatus();

private:
	std::string ModelPath; 
	std::string ModelInputSignature;
	std::string ModelOutputSignature;
	std::string ModelTagName;

	TF_Buffer* run_options = TF_NewBufferFromString("", 0);
	TF_SessionOptions* session_options = TF_NewSessionOptions();
	TF_Graph* graph = TF_NewGraph();
	TF_Session* session;


	TF_Operation* op_input;
	TF_Operation* op_output;
	std::array<TF_Output, 1> input_ops;
	std::array<TF_Output, 1> output_ops;

	bool FLAG_MODEL_LOADED;
};

