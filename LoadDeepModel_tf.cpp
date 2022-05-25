#include "LoadDeepModel_tf.h"
#include <iostream>
#include <array>

LoadDeepModel::LoadDeepModel(std::string str_model_path, std::string str_model_input_signature, std::string str_model_output_signature, std::string str_model_tag_name)
{
	ModelPath = str_model_path;
	ModelInputSignature = str_model_input_signature;
	ModelOutputSignature = str_model_output_signature;
	ModelTagName = str_model_tag_name;
	loadModel();
}

void LoadDeepModel::setModel(std::string model_path, std::string str_model_input_signature, std::string str_model_output_signature, std::string str_model_tag_name)
{
	ModelPath = model_path;
	ModelInputSignature = str_model_input_signature;
	ModelOutputSignature = str_model_output_signature;
	ModelTagName = str_model_tag_name;
	loadModel();
}

bool LoadDeepModel::loadModel()
{
	FLAG_MODEL_LOADED = false;

	//std::array<char const*, 1> tags{"serve"}; 
 	std::array<char const*, 1> tags{ ModelTagName.c_str() };
	 
 	session = TF_LoadSessionFromSavedModel(session_options, run_options, ModelPath.c_str(), \
		tags.data(), tags.size(), graph, nullptr, status);

	// if the session is not loaded
	if (TF_GetCode(status) != TF_OK)
		return false;

	op_input = TF_GraphOperationByName(graph, ModelInputSignature.c_str());
	// failed to find
	if (op_input == nullptr)
		return false;
	op_output = TF_GraphOperationByName(graph, ModelOutputSignature.c_str());
	// failed to find
	if (op_output == nullptr)
		return false;

	// for batch
	input_ops = { TF_Output{ op_input, 0 } };
	output_ops = { TF_Output{ op_output, 0 } };
	FLAG_MODEL_LOADED = true;
	return true;
}

bool LoadDeepModel::isModelLoaded()
{
	return FLAG_MODEL_LOADED;
}

cv::Mat LoadDeepModel::predict(cv::Mat* mat, float threshold)
{
	cv::Mat matCopied = mat->clone();
	if (!FLAG_MODEL_LOADED) return matCopied;

	// input tensor's dimention
	int batch_size = 1;
	std::array<int64_t, 4> const dims{ static_cast<int64_t>(batch_size), static_cast<int64_t>(matCopied.rows), static_cast<int64_t>(matCopied.cols), static_cast<int64_t>(matCopied.channels()) };

	// norm [0,1]
	matCopied /= 255;
	void* data = (void*)matCopied.data;
	std::size_t const ndata = batch_size * matCopied.rows * matCopied.cols * matCopied.channels() * TF_DataTypeSize(TF_FLOAT);

	auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII
	
	auto* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), dims.size(), data, ndata, deallocator, nullptr);
	std::array<TF_Tensor*, 1> input_values{ input_tensor };
	std::array<TF_Tensor*, 1 > output_values{};

	TF_SessionRun(session, run_options, input_ops.data(), input_values.data(), input_ops.size(), output_ops.data(), output_values.data(), output_ops.size(), nullptr, 0, nullptr, status);
	
	auto* output_tensor = static_cast<std::array<float, 1> *>(TF_TensorData(output_values[0]));

	cv::Mat output(cv::Size(matCopied.rows, matCopied.cols), CV_32FC1, output_tensor->data());
	cv::Mat output2(output.rows, output.cols, CV_8UC1);

	for (int h = 0; h < output.rows; ++h)
	{
		for (int w = 0; w < output.cols; ++w)
		{
			float pixel = *output.ptr<float>(h, w);
			if (pixel < threshold)
				//*output.ptr<float>(h, w) = 0;
				*output2.ptr<uchar>(h, w) = 0;
			else
				//*output.ptr<float>(h, w) = 1;
				*output2.ptr<uchar>(h, w) = 255;
		}
	}
	return output2;
}

cv::Mat LoadDeepModel::getResizedImageWithMask(cv::Mat palette, cv::Mat mask, int scaleRate)
{
 	if (palette.type() != CV_8UC3 || palette.empty() || mask.type() != CV_8UC1)
		return palette;
	if (palette.rows != mask.rows || palette.cols != mask.cols)
	{
		return palette;
	}
	cv::Mat pal = palette.clone();
	cv::imshow("pred", pal);
	cv::waitKey(0);
	cv::destroyAllWindows();

	int scaleR = scaleRate;
	if (scaleR <= 0 || scaleR % 2 != 0 || scaleR >8) scaleR = 2;

	cv::Mat output(pal.rows * scaleR, pal.cols * scaleR, CV_8UC3);

	uchar* pPaletteRow;
	uchar* pMaskRow;
	uchar r;
	uchar g;
	uchar b;

	// modify red pixel += 60
	for (int h = 0; h < pal.rows; ++h)
	{
		pPaletteRow = pal.ptr<uchar>(h);
		pMaskRow = mask.ptr<uchar>(h);
		for (int w = 0; w < pal.cols; ++w)
		{
			if (pMaskRow[w] == 255)
			{
				r = pPaletteRow[w * 3 + 2];
				r += 60;
				pPaletteRow[w * 3 + 2] = r;

			}
		}
	}

	// scale pixel
	for (int h = 0; h < pal.rows; ++h)
	{
		pPaletteRow = pal.ptr<uchar>(h);
		for (int w = 0; w < pal.cols; ++w)
		{
			b = pPaletteRow[w * 3 + 0];
			g = pPaletteRow[w * 3 + 1];
			r = pPaletteRow[w * 3 + 2];
			for (int mh = h * scaleR; mh < h * scaleR + scaleR; ++mh)
			{
				for (int mw = w * scaleR; mw < w * scaleR + scaleR; ++mw)
				{
					*output.ptr<cv::Vec3b>(mh, mw) = cv::Vec3b(b, g, r);
					//output.at<cv::Vec3b>(mh, mw) = cv::Vec3b(b, g, r);
				}
			}
		}
	}

	return output;
}
