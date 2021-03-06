/*
 * UML-diag Generate At https://yuml.me/diagram/scruffy/class/draw
 * Content As : [ActorComponent||+operation()]&lt;0..*-uses&lt;&gt;[ActorDecorator|-actor_component_vec|+operation();+addcomponent(){bg:orange}], [ActorComponent]^-.-[ActorDecorator], [ActorComponent]^-.-[IActor||+init(...);+operate();+operation()], [IActor]^-.-[Actor_etc..|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[IActor]^-.-[Actor_2nd|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()], [IActor]^-.-[VideoReader|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[ActorDecorator]^[ActorDecorator_1st], [ActorFactory||+createActorComponent(...){bg:green}]^[ConcreteActorFactory||+createActorComponent(...)], [ConcreteActorFactory]creates-.-&gt;[IActor],[user||main()]uses-&gt;[ActorFactory],[user||main()]uses-&gt;[ActorDecorator],[note: The ActorDecorator is an interface for the process such as Slideshow etc...]-.->[ActorDecorator],[note: The IActor is an interface for smallest unit in a process such as Camera  DataProcess NNInference etc...]-.->[IActor]
 * Coding Style Following: https://www.kernel.org/doc/html/v4.10/process/coding-style.html 
 * Coding Style Following: https://en.cppreference.com
 */

// std headers
#include <iostream>
#include <string>
#include <vector> 
#include <thread>
#include <chrono>

// linux
#include <dirent.h>

// opencv headers
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

// third party headers
#include "Queue.h" 

// GTI headers
#include "GTILib.h"
#include "GtiClassify.h"
#include "Classify.hpp"

// CaffeInferencer
#include <caffe/caffe.hpp>

// For cv human detector
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

// For GetTickCount
#include <time.h>


#define CAFFE_IMG_INPUT_FLAG 0
#define DEBUG_INFO_FLAG 0

#define ImageQueue Queue<cv::Mat>
#define StreamQueue Queue<std::ostream>
#define DATA_PATH  "./";

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)


double GetTickCount(void) 
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now))
    return 0;
  return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

class BoxData {
 public:
  int label_;
  bool difficult_;
  float score_;
  vector<float> box_;
};

typedef struct {
	// genernal
	char *name;
	// ImageShower
	char *disp_name;
	// VideoReader
	char *source;
	int width;
	int height;
	// GTIInferencer
	int gti_input_image_width;
	int gti_input_image_height;
	int gti_input_image_channel;
	int gti_device_type;
	char *gti_device_name;
	char *gti_nn_weight_name;
	char *gti_user_config;
	// GTIFcClassifier
	char *gti_fc_name;
	char *gti_fc_label;
	// CaffeInferencer
	char *caffe_model_file;
	char *caffe_trained_file;
	char *caffe_mean_file;
	char *caffe_label_file;
} ACTOR_ARG;

typedef struct {
	int top;
	int left;
	int width;
	int height;
	string label;
	float score;
} BBOX_META;

class IFrame{
	public:
		IFrame(): buffer(nullptr){}
		~IFrame(){
			// release resurce
			cout << "IFrame Release Image" << "\n";
			cout << "IFrame Release Image img.size:" << img.size() << "\n";
			for (auto image: img)
			{
				cout << "IFrame image.deallocate" << "\n";
				assert(image.data);
				image.deallocate();
				image.release();
				assert( !image.data);	// check image is NULL
			}
			img.clear();

			for (auto box: pred_vec)
			{
				delete box;
				box = nullptr;
			}
			pred_vec.clear();
			
			cout << "IFrame Release buffer" << "\n";
			if (buffer != nullptr)
			{
				delete buffer;
				buffer = nullptr;
			}
		}
	public:
		std::vector<cv::Mat> img;
		std::vector<BBOX_META*> pred_vec;
		int buffer_length;
		float *buffer;
};

#define FrameQueue Queue<IFrame*>

class ActorComponent {
	public:
		virtual ~ActorComponent(){}
		virtual void operation() = 0;
};

class IActor: public ActorComponent{
	public:
		virtual ~IActor(){}
		virtual void init() = 0;
		virtual void operate() = 0;
		virtual void operation() = 0;
};

class VideoReader: public IActor{
	public:
		VideoReader(ACTOR_ARG &actor_arg, FrameQueue &out_queue) {
			_out_queue = &out_queue;
			_demofile = actor_arg.source;
			_width1 = actor_arg.width;
			_height1 = actor_arg.height;
		}
		~VideoReader() {
			// release resurce
			while (_thread.joinable())
				_thread.join();

			_img.release();
			_img1.release();
			assert( !_img.data);
			assert( !_img1.data);

			cout << "del_VideoReader" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "VideoReader.init" << "\n";


			std::cout << "Open Video file " << _demofile << " ...\n";
			_cap.open(_demofile);

			if ( !_cap.isOpened())
			{
				_cap.release();
				std::cerr << "Video Clip File " << _demofile << " is not opened!\n";
				return;
			}

			_width  = _cap.get(CV_CAP_PROP_FRAME_WIDTH);
			_height  = _cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			_frame_count  = _cap.get(CV_CAP_PROP_FRAME_COUNT);
			int video_cc = _cap.get(CV_CAP_PROP_FOURCC);
			_video_codec  = format("%c%c%c%c", video_cc & 255, (video_cc >> 8) & 255, (video_cc >> 16) & 255, (video_cc >> 24) & 255);
			_pause_flag = 0;

			_frame_num = 0;
		}
		/*virtual*/
		void operate() {
			cout << "VideoReader.op " << "\n";
			int i = 0;
			while (1)
			{
				i++;
				if (i > 500)
					_pause_flag = 1;
				if (_pause_flag == 0)
				{
					bool cap_read_false;
					// reset if video playing ended
					if (_frame_num == _frame_count)
					{
						_cap.set(CV_CAP_PROP_POS_FRAMES, 0);
						_frame_num = 0;
					}
					// OpenCV uses BGR color space (imread, VideoCapture).
					_cap.read(_img);
					_frame_num++;


					std::cout << "_video_codec: " << _video_codec <<  "  ...\n";
					std::cout << "_frame_num: " << _frame_num << "/" << _frame_count << "  ...\n";
					cv::resize(_img, _img1, cv::Size(_width1, _height1));

					IFrame* iframe = new IFrame();
					//iframe->img.push_back(cv::Mat(_img1));
					iframe->img.push_back(_img1.clone());

					_out_queue->push(iframe);
					std::this_thread::sleep_for(std::chrono::milliseconds(100));

					// release resurce
					_img.release();
					_img1.release();
					assert( !_img.data);
					assert( !_img1.data);
				}
				else
					break;
			}
		}

		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}

	private:
		FrameQueue *_out_queue;
		int _width1;
		int _height1;
		string _demofile;

		thread _thread;
		cv::VideoCapture _cap;
		int _width;
		int _height;
		int _frame_count;
		string _video_codec;
		bool _pause_flag;
		cv::Mat _img;
		cv::Mat _img1;
		int _frame_num;
};

class ImageReader: public IActor{
	public:
		ImageReader(ACTOR_ARG &actor_arg, FrameQueue &out_queue) {
			_out_queue = &out_queue;
			_demo_dir = actor_arg.source;
			_width1 = actor_arg.width;
			_height1 = actor_arg.height;
		}
		~ImageReader() {
			// release resurce
			while (_thread.joinable())
				_thread.join();

			_img.release();
			_img1.release();
			assert( !_img.data);
			assert( !_img1.data);

			cout << "del_ImageReader" << '\n';
		}
		/*virtual*/
		void init() {
			std::cout << "ImageReader.init" << "\n";
			
			_pause_flag = 0;

			_frame_num = 0;
		}
		/*virtual*/
		void operate() {
			cout << "ImageReader.op " << "\n";
			int i = 0;
        		_dir = opendir(_demo_dir.c_str());
			if (nullptr == _dir)
			{
				// could not open directory
				std::cout << "Open " << _demo_dir.c_str() << " folder error.\n";
				return;
			}

			while (1)
			{
				i++;
				//if (i > 50)
				//	_pause_flag = 1;
				if (_pause_flag == 0)
				{
					if ((_ent = readdir (_dir)) != nullptr)
					{
						string image_filename;
						
						string filename = _ent->d_name;

						if ((filename.size() == 0) || (filename == ".") || (filename == ".."))
						{
							continue;
						}
            					image_filename = _demo_dir + "/" + filename;

						_img = cv::imread(image_filename, -1);
						if (_img.empty())
						{
						    //std::cout << "File not found." << std::endl;
						    continue;
						}
            					cv::resize(_img, _img1, cv::Size(_width1, _height1));	

						IFrame* iframe = new IFrame();
						iframe->img.push_back(cv::Mat(_img1));
						//iframe->img.push_back(_img1.clone());
						_out_queue->push(iframe);
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
					}
					else
					{
        					closedir (_dir);
        					_dir = opendir(_demo_dir.c_str());
						continue;
					}
					// release resurce
					_img.release();
					_img1.release();
					assert( !_img.data);
					assert( !_img1.data);
				}
				else
					break;
			}
		}

		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}

	private:
		FrameQueue *_out_queue;
		int _width1;
		int _height1;
		string _demo_dir;

		DIR *_dir;
		struct dirent *_ent;

		thread _thread;

		bool _pause_flag;
		cv::Mat _img;
		cv::Mat _img1;
		int _frame_num;
};

class ImageShower: public IActor{
	public:
		ImageShower(ACTOR_ARG &actor_arg, FrameQueue &in_queue) {
			_in_queue = &in_queue;
			_window_name = actor_arg.disp_name;
			_display_width = 480;
			_display_height = 480;
		}
		~ImageShower() {
        		// release resurce
			while (_thread.joinable())
				_thread.join();
                        
			_img.release();
			assert( !_img.data);

			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			cout << "del_ImageShower" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "ImageShower.init" << "\n";
		}
		/*virtual*/
		void operate() {
			cout << "ImageShower.op " << "\n";
			while (1)
			{
				cout << "ImageShower Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					_iframe = _in_queue->pop();

					_img = _iframe->img[0];
					cout << "ImageShower prepare show" << "\n";
					cv::resize(_img, _img, cv::Size(_display_width, _display_height));
					int scale_x = _display_width / 224;
					int scale_y = _display_height / 224;
					for (BBOX_META* box: _iframe->pred_vec)
					{
						std::stringstream ss;
						ss << box->label << ":" << box->score;
						if (box->top < 0)
							box->top = 0;
						if (box->left < 0)
							box->left = 0;
						if (box->top + box->height > 224)
							box->height = 220 - box->top;
						if (box->width + box->left > 224)
							box->width = 220 - box->left;
							
						//cv::String info(ss.str());
						int font_face = FONT_HERSHEY_SIMPLEX;
						double font_scale = 0.4;
						int thickness = 1;
						cv::Rect rect(box->left * scale_x, box->top * scale_y, box->width * scale_x, box->height * scale_y);
						cv::Rect rect1(box->left * scale_x, box->top * scale_y, box->width * scale_x, 10 * scale_y);
						cv::rectangle(_img, rect1, cv::Scalar(0, 255, 0), -1, 1, 0);
						cv::putText(_img, ss.str(), cv::Point(rect1.x + 5, rect1.y + 15), font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
						cv::rectangle(_img, rect, cv::Scalar(0, 255, 0), 1, 1, 0);
					}
					cv::imshow(_window_name, _img);
					cv::waitKey(1);

					cout << "ImageShower _img.deallocate" << "\n";
        				// release resurce
					_img.deallocate();
					_img.release();
					assert( !_img.data);

        				// release resurce
					delete _iframe;
					_iframe = nullptr;
					assert( !_iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}
	private:
		FrameQueue* _in_queue;
		string _window_name;
		int _display_width;
		int _display_height;
		thread _thread;

		bool _pause_flag;
		cv::Mat _img;
		IFrame* _iframe;
};

class DetectedImageShower: public IActor{
	public:
		DetectedImageShower(ACTOR_ARG &actor_arg, FrameQueue &in_queue) {
			_in_queue = &in_queue;
			_window_name = actor_arg.disp_name;
		}
		~DetectedImageShower() {
        		// release resurce
			while (_thread.joinable())
				_thread.join();
                        
			_img.release();
			assert( !_img.data);

			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			cout << "del_DetectedImageShower" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "DetectedImageShower.init" << "\n";
		}
		/*virtual*/
		void operate() {
			cout << "DetectedImageShower.op " << "\n";
			while (1)
			{
				cout << "DetectedImageShower Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					_iframe = _in_queue->pop();

					cout << "DetectedImageShower _img.deallocate" << "\n";

        				// release resurce
					_img.deallocate();
					_img.release();
					assert( !_img.data);

					_img = _iframe->img[0];
					//cv::imshow(_window_name, _img);
					//cv::waitKey(1);

        				// release resurce
					delete _iframe;
					_iframe = nullptr;
					assert( !_iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}
	private:
		FrameQueue* _in_queue;
		string _window_name;
		thread _thread;

		bool _pause_flag;
		cv::Mat _img;
		IFrame* _iframe;
};

class GTIInferencer: public IActor{
	public:
		GTIInferencer(ACTOR_ARG &actor_arg, FrameQueue &in_queue, FrameQueue &out_queue) {
			_in_queue = &in_queue;
			_out_queue = &out_queue;

			_gti_input_image_width = actor_arg.gti_input_image_width;
			_gti_input_image_height = actor_arg.gti_input_image_height;
			_gti_input_image_channel = actor_arg.gti_input_image_channel;

			_gti_device_type = actor_arg.gti_device_type;
			_gti_device_name = actor_arg.gti_device_name;
			_gti_nn_weight_name = actor_arg.gti_nn_weight_name;
			_gti_user_config = actor_arg.gti_user_config;

		}
		~GTIInferencer() {
			while (_thread.joinable())
				_thread.join();

			if (_output_image_buffer != nullptr)
			{
				delete _output_image_buffer;
				_output_image_buffer = nullptr;
			}
			if (_input_image_buffer != nullptr)
			{
				delete _input_image_buffer;
				_input_image_buffer = nullptr;
			}

			// Close device
			GtiCloseDevice(_device);
			// Release device
			GtiDeviceRelease(_device);
			cout << "del_GTIInferencer" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "GTIInferencer.init" << "\n";
			// Create GTI device
			_device = GtiDeviceCreate(_gti_device_type, (char *)_gti_nn_weight_name.c_str(), (char *)_gti_user_config.c_str());

			// Open device
			GtiOpenDevice(_device, (char *)_gti_device_name.c_str());	

			// Initialize GTI SDK
			if (!GtiInitialization(_device))
			{
				cout << "GTIInferencer initialization failed." <<"\n";	
			}

			// Allocate memory for sample code use
			_output_image_buffer = nullptr;
			_input_image_buffer = nullptr;
			_output_image_buffer = new float[GtiGetOutputLength(_device)];	
			if ( !_output_image_buffer)
			{
			}
			_input_image_buffer = new unsigned char[_gti_input_image_width * _gti_input_image_height * _gti_input_image_channel];
			if ( !_input_image_buffer)
			{
				cout << "GTIInferencer allocation (_input_image_buffer) failed." << "\n";
			}
			_input_image_float_buffer = new float[_gti_input_image_width * _gti_input_image_height * _gti_input_image_channel];
			if ( !_input_image_float_buffer)
			{
				cout << "GTIInferencer allocation (_input_image_float_buffer) failed." << "\n";
			}
						
		}
		/*virtual*/
		void operate() {
			cout << "GTIInferencer.op " << "\n";
			int input_length = _gti_input_image_width * _gti_input_image_height * _gti_input_image_channel;
			while (1)
			{
				cout << "GTIInferencer Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					IFrame *iframe = _in_queue->pop();
					
					cv::Mat img(iframe->img[0]);
					cv::Mat output(iframe->img[0]);
					std::vector<cv::Mat> input_image;
					//cv::Mat input_image;
					// converts input image to the float point 3 channel image.
					cout << "GTIInferencer img CV_8U convertTo CV_32F !" << "\n";
					//img.convertTo(input_image, CV_32F);
					cnnSvicProc32FC3(img, &input_image);
					// converts 32 bit float format image to 8 bit integer format image. 
					cout << "GTIInferencer img 32F convertTo 8Byte !" << "\n";
					cvt32FloatTo8Byte((float *)input_image[0].data, (uchar *)_input_image_buffer, _gti_input_image_width, _gti_input_image_height, _gti_input_image_channel);

					cout << "GTIInferencer Create _nn_output_buffer !" << "\n";					
					float *_nn_output_buffer = new float[GtiGetOutputLength(_device)];

					cout << "GTIInferencer GtiHandleOneFrameFloat !" << "\n";
					int ret = GtiHandleOneFrameFloat(_device, _input_image_buffer, input_length,  _nn_output_buffer, GtiGetOutputLength(_device));
					if ( !ret)
					{
						cout << "GTIInferencer Handle one frame error!" << "\n";
						return;
					}
					
					img.release();
					output.release();
					for (cv::Mat image: input_image)
						image.release();

					iframe->buffer = _nn_output_buffer;
					iframe->buffer_length = GtiGetOutputLength(_device);
					
					_out_queue->push(iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}

		//====================================================================
		// Function name: void cnnSvicProc32FC3(const cv::Mat& img,
		//                          std::vector<cv::Mat>* input_channels)
		// This function converts input image to the float point 3 channel image.
			//
		// Input: const cv::Mat& img - input image.
		//        std::vector<cv::Mat>* input_channels - output buffer to store
		//              float point 3 channel image.
		// return: none.
		//====================================================================
		void cnnSvicProc32FC3(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
		{
		    cv::Mat sample;
		    cv::Mat sample_resized;
		    cv::Mat sample_byte;
		    cv::Mat sample_normalized;

		    const int num_channels = _gti_input_image_channel;
		    int width = _gti_input_image_width;
		    int height = _gti_input_image_height;
		    float *input_data = _input_image_float_buffer;
		    if (input_data == NULL)
		    {
			std::cout << "Failed allocat memory for input_data!" << std::endl;
			return;
		    }

		    for (int i = 0; i < num_channels; ++i)
		    {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		    }

		    sample = img;
		    switch (img.channels())
		    {
		    case 1:
			if (num_channels == 3)
			{
			    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
			}
			break;
		    case 3:
			if (num_channels == 1)
			{
			    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
			}
			break;
		    case 4:
			if (num_channels == 1)
			{
			    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
			}
			else if (num_channels == 3)
			{
			    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
			}
			break;
		    default:
			break;
		    }

		    sample_resized = sample;

		    if (num_channels == 3)
		    {
			sample_resized.convertTo(sample_byte, CV_32FC3);
		    }
		    else
		    {
			sample_resized.convertTo(sample_byte, CV_32FC1);
		    }

		    cv::subtract(sample_byte, cv::Scalar(0., 0., 0.), sample_normalized);

		    /* This operation will write the separate BGR planes directly to the
		     * input layer of the network because it is wrapped by the cv::Mat
		     * objects in input_channels. */
		    cv::split(sample_normalized, *input_channels);
		}

		//====================================================================
		// Function name: void cvt32FloatTo8Byte(float *InBuffer, uchar *OutBuffer,
		//                      int Width, int Height, int Channels)
		// This function converts 32 bit float format image to 8 bit integer
		// format image.
		//
		// Input: float *InBuffer - input 32 bits/pixel data buffer.
		//        uchar *OutBuffer - output buffer to store 8 bits/pixel image data.
		//        int Width - image width in pixel
		//        int Height - image height in pixel
		//        int Channels - image channels
		// return: none.
		//====================================================================
		void cvt32FloatTo8Byte(float *InBuffer, uchar *OutBuffer,
		                  int Width, int Height, int Channels)
		{
		    uchar *pOut = OutBuffer;
		    float *pIn = InBuffer;
		    if (pIn == NULL || pOut == NULL)
		    {
		        std::cout << "cvt32FloatTo8Byte: null pointer!" << std::endl;
		        return;
		    }
		
		    for (int i = 0; i < Channels; i++)
		    {
		        for (int j = 0; j < Height; j++)
		        {
		            for (int k = 0; k < Width; k++)
		            {
		                *pOut++ = (uchar)*pIn++;
		            }
		        }
		    }
		}
		
	private:
		FrameQueue *_in_queue;
		FrameQueue *_out_queue;

		int _gti_input_image_width;
		int _gti_input_image_height;
		int _gti_input_image_channel;

		int _gti_device_type;
		string _gti_device_name;
		string _gti_nn_weight_name;
		string _gti_user_config;


		thread _thread;

		GtiDevice *_device;
		float *_output_image_buffer;
		float *_input_image_float_buffer;
		unsigned char *_input_image_buffer; 

		bool _pause_flag;
};

class GTIFcClassifier: public IActor{
	public:
		GTIFcClassifier(ACTOR_ARG &actor_arg, FrameQueue &in_queue) {
			_in_queue = &in_queue;

			_gti_fc_name = actor_arg.gti_fc_name;
			_gti_fc_label = actor_arg.gti_fc_label;
		}
		~GTIFcClassifier() {
			while (_thread.joinable())
				_thread.join();
        		// release resurce
			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			// Release FC
			GtiClassifyRelease(_gti_classifier);
			cout << "del_GTIFcClassifier" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "GTIFcClassifier.init" << "\n";
			cout << "GTIFcClassifier initialization FC." <<"\n";
			_gti_classifier = GtiClassifyCreate(_gti_fc_name.c_str(), _gti_fc_label.c_str());
						
		}
		/*virtual*/
		void operate() {
			cout << "GTIFcClassifier.op " << "\n";
			while (1)
			{
				cout << "GTIFcClassifier Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					_iframe = _in_queue->pop();
					float *_nn_output_buffer = _iframe->buffer;
					
					cout << "GTIFcClassify GtiClassifyFC !" << "\n";
					GtiClassifyFC(_gti_classifier, _nn_output_buffer, 5);

					/* Print the top N predictions. */	
					for (int i = 0; i < 5; ++i)
					{
						char *ptext = GetPredicationString(_gti_classifier, i);
						cout << "GTIFcClassifier ptext: " << ptext << "\n";
					}

					cout << "GTIFcClassifier Release iframe !" << "\n";

        				// release resurce
					delete _iframe;
					_iframe = nullptr;
					assert( !_iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}

		
	private:
		FrameQueue *_in_queue;

		string _gti_fc_name;
		string _gti_fc_label;

		thread _thread;

		Classify *_gti_classifier;

		bool _pause_flag;
		IFrame *_iframe;
};

class CaffeInferencer: public IActor{
	public:
		CaffeInferencer(ACTOR_ARG &actor_arg, FrameQueue &in_queue, FrameQueue &out_queue) {
			_in_queue = &in_queue;
			_out_queue = &out_queue;

			_caffe_model_file = actor_arg.caffe_model_file;
			_caffe_trained_file = actor_arg.caffe_trained_file;
			_caffe_mean_file = actor_arg.caffe_mean_file;
			_caffe_label_file =  actor_arg.caffe_label_file;
		}
		~CaffeInferencer() {
			while (_thread.joinable())
				_thread.join();
        		// release resurce
			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			// Release Classifier
			cout << "del_CaffeInferencer" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "CaffeInferencer.init" << "\n";
			cout << "CaffeInferencer initialization FC." <<"\n";

			Caffe::set_mode(Caffe::CPU);
			_net.reset(new Net<float>(_caffe_model_file.c_str(), TEST));
			_net->CopyTrainedLayersFrom(_caffe_trained_file.c_str());
			_input_layer = _net->input_blobs()[0];
			//input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

			/* Load the binaryproto mean file. */
			/* Load labels. */
						
		}
		/*virtual*/
		void operate() {
			cout << "CaffeInferencer.op " << "\n";
			while (1)
			{
				cout << "CaffeInferencer Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					cout << "CaffeInferencer  !" << "\n";
					_iframe = _in_queue->pop();
					
					
#if CAFFE_IMG_INPUT_FLAG
					// Wrap the input layer of the network in separate cv::Mat objects
					cout << "CaffeInferencer  Wrap the input layer !" << "\n";
					std::vector<cv::Mat> input_channels;
					int width = _input_layer->width();
					int height = _input_layer->height();
					cout << "CaffeInferencer get mutable_cpu_data !" << "\n";
					float *input_data = _input_layer->mutable_cpu_data();
					for (int i = 0; i < _input_layer->channels(); ++i) {
						cv::Mat channel(height, width, CV_32FC1, input_data);
						cout << "CaffeInferencer Wrap the input layer !" << "\n";
						input_channels.push_back(channel);
						input_data += width * height;
					}

					cout << "CaffeInferencer  Preprocessing  !" << "\n";
					cv::Mat output(_iframe->img[0]);
					//std::vector<cv::Mat> input_image;
					cv::Mat input_image;
					// converts input image to the float point 3 channel image.
					//cnnSvicProc32FC3(img, &input_image);
					//cout << "CaffeInferencer img CV_8U convertTo CV_32F !" << "\n";
					output.convertTo(input_image, CV_32F);

					cout << "CaffeInferencer  Write image to net  !" << "\n";
					/* This operation will write the separate BGR planes directly to the
					 * input layer of the network because it is wrapped by the cv::Mat
					 * objects in input_channels. 
					 */
					cv::split(input_image, input_channels);
#else
					float *nn_output_buffer = _iframe->buffer;
					float *input_data = _input_layer->mutable_cpu_data();
					cout << "CaffeInferencer  feed buffer to Caffe  !" <<  "\n";
					for (int i = 0;i < _iframe->buffer_length; ++i) {
						input_data[i] = nn_output_buffer[i]; 
					}
#endif
					cout << "CaffeInferencer Forward  !" << "\n";
					// Caffe Inference
					_net->Forward();

					// Copy the output layer to a std::vector
					Blob<float> *output_layer = _net->output_blobs()[0];
					const float *begin = output_layer->cpu_data();
					const float *end = begin + output_layer->channels();
					std::vector<float> net_result = std::vector<float>(begin, end);
	
					cout << "CaffeInferencer net_result.size: " << net_result.size() << "\n";

					float *_nn_output_buffer = new float[net_result.size()];
					//memcpy(_nn_output_buffer, begin, net_result.size());
					std::copy(net_result.begin(), net_result.end(), _nn_output_buffer);
					_iframe->buffer = _nn_output_buffer;
					_iframe->buffer_length = net_result.size();
					_out_queue->push(_iframe);
					
#if CAFFE_IMG_INPUT_FLAG
					output.release();
					input_image.release();
#endif
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}
		
	private:
		FrameQueue *_in_queue;
		FrameQueue *_out_queue;

		string _caffe_model_file;
		string _caffe_trained_file;
		string _caffe_mean_file;
		string _caffe_label_file;

		std::shared_ptr<Net<float> > _net;
		Blob<float>* _input_layer;

		thread _thread;


		bool _pause_flag;
		IFrame *_iframe;
};

class YOLODetector: public IActor{
	public:
		YOLODetector(ACTOR_ARG &actor_arg, FrameQueue &in_queue, FrameQueue &out_queue) {
			_in_queue = &in_queue;
			_out_queue = &out_queue;

			_side = 7;
			_num_objects = 2;
			_num_classes = 20;
			_sqrt = true;
			_constriant = true;
			_score_type = 0;
			_nms = 0.4;
			_score_threshold = 0.2;

			_label_map_name = "/root/workspaces/vgg_yolo/label_map.txt";

		}
		~YOLODetector() {
			while (_thread.joinable())
				_thread.join();

			_img.release();
			assert( !_img.data);

        		// release resurce
			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			// Release YOLODetector
			cout << "del_YOLODetector" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "YOLODetector.init" << "\n";
			cout << "YOLODetector initialization FC." <<"\n";	
			cout << "YOLODetector load label." <<"\n";	
			ifstream read(_label_map_name);
			for (std::string line; std::getline(read, line); )
			{
				int pos = line.find(" ");
				int label_id = std::stoi(line.substr(pos));
				string label_name = line.substr(0, pos);	
				_label_map[label_id] = label_name;
				
			}
			cout << "YOLODetector label info." <<"\n";	
			for (const auto& map: _label_map)
			{
				cout << "key:" << map.first << " label name:" << map.second << "\n";
			}
			
		}
		/*virtual*/
		void operate() {
			cout << "YOLODetector.op " << "\n";
			string window_name = "test";
			while (1)
			{

				cout << "YOLODetector Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					cout << "YOLODetector  !" << "\n";
					_iframe = _in_queue->pop();

        				// release resurce
					_img.deallocate();
					_img.release();
					assert( !_img.data);

					_img = _iframe->img[0];

					std::vector<float> net_result(_iframe->buffer, _iframe->buffer + _iframe->buffer_length);
						
					cout << "YOLODetector input_result.size: " << net_result.size() << "\n";
					for (int i = 0; i < 50; i++)
					{
						cout << net_result[i] << " ";
					}
					cout << "YOLODetector show_det exp." << "\n";
					int locations = _side * _side;

    					map<int, vector<BoxData> > pred_boxes;
					GetPredBox(_side, _num_objects, _num_classes, net_result, &pred_boxes, _sqrt, _constriant, _score_type, _nms);

					int pred_count = 0;
					cout << "YOLODetector Result !" << "\n";
					for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) 
					{
						cout << "YOLODetector BoxData " <<  pred_count << " !" << "\n";
						int label = it->first;
						vector<BoxData>& p_boxes = it->second;
						for (int b = 0; b < p_boxes.size(); ++b)
						{
							if (p_boxes[b].score_ > _score_threshold)
							{

								//int centor_x = p_boxes[b].box_[0] * 224 - p_boxes[b].box_[0] * (224 / _side);
								//int centor_y = p_boxes[b].box_[1] * 224 - p_boxes[b].box_[1] * (224 / _side);
								int centor_x = p_boxes[b].box_[0] * 224;
								int centor_y = p_boxes[b].box_[1] * 224;

								int width = p_boxes[b].box_[2] * 224;
								int height = p_boxes[b].box_[3] * 224;
								int top = centor_y - height / 2;
								int left = centor_x - width / 2;
#if DEBUG_INFO_FLAG	
								cout << "YOLODetector p_boxes[b].label: " << p_boxes[b].label_ << "\n";
								cout << "YOLODetector p_boxes[b].score: " << p_boxes[b].score_ << "\n";
								cout << "YOLODetector p_boxes[b].top: " << top << "\n";
								cout << "YOLODetector p_boxes[b].left: " << left << "\n";
								cout << "YOLODetector p_boxes[b].width: " << width << "\n";
								cout << "YOLODetector p_boxes[b].height: " << height << "\n";
#endif
								BBOX_META *bbox_data = new BBOX_META{top, left, width, height, _label_map[label], p_boxes[b].score_};
								_iframe->pred_vec.push_back(bbox_data); 
							}
						}
						++pred_count;
					}
					


					_out_queue->push(_iframe);
					//cout << "YOLODetector Release iframe !" << "\n";
        				// release resurce
					//delete _iframe;
					//_iframe = nullptr;
					//assert( !_iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}

		
		void ApplyNms(const vector<BoxData> &boxes, vector<int> *idxes, float threshold) {
			map<int, int> idx_map;
			for (int i = 0; i < boxes.size() - 1; ++i) {
				if (idx_map.find(i) != idx_map.end()) {
					continue;
				}
				vector<float> box1 = boxes[i].box_;
				for (int j = i + 1; j < boxes.size(); ++j) {
					if (idx_map.find(j) != idx_map.end()) {
						continue;
					}
					vector<float> box2 = boxes[j].box_;
					float iou = Calc_iou(box1, box2);
					if (iou >= threshold) {
						idx_map[j] = 1;
					}
				}
			}
			for (int i = 0; i < boxes.size(); ++i) {
				if (idx_map.find(i) == idx_map.end()) {
					idxes->push_back(i);
				}
			}
		}

		float Overlap(float x1, float w1, float x2, float w2) {
			float left = std::max(x1 - w1 / 2, x2 - w2 / 2);
			float right = std::min(x1 + w1 / 2, x2 + w2 / 2);
			return right - left;
		}

		float Calc_iou(const vector<float> &box, const vector<float> &truth) {
			float w = Overlap(box[0], box[2], truth[0], truth[2]);
			float h = Overlap(box[1], box[3], truth[1], truth[3]);
			if (w < 0 || h < 0) return 0;
			float inter_area = w * h;
			float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
			return inter_area / union_area;
		}

		void GetPredBox(int side, int num_object, int num_class, const std::vector<float> &input_data,
		            map<int, vector<BoxData> > *pred_boxes, bool use_sqrt, bool constriant, 
		            int score_type, float nms_threshold) {
			vector<BoxData> tmp_boxes;
			int locations = pow(side, 2);
			for (int i = 0; i < locations; ++i) {
				int pred_label = 0;
				float max_prob = input_data[i];
				for (int j = 1; j < num_class; ++j) {
					int class_index = j * locations + i;   
					if (input_data[class_index] > max_prob) {
						pred_label = j;
						max_prob = input_data[class_index];
					}
				}
				if (nms_threshold < 0) {
					if (pred_boxes->find(pred_label) == pred_boxes->end()) {
						(*pred_boxes)[pred_label] = vector<BoxData>();
					}
				}
				// LOG(INFO) << "pred_label: " << pred_label << " max_prob: " << max_prob; 
				int obj_index = num_class * locations + i;
				int coord_index = (num_class + num_object) * locations + i;
				for (int k = 0; k < num_object; ++k) {
					BoxData pred_box;
					float scale = input_data[obj_index + k * locations];
					pred_box.label_ = pred_label;
					if (score_type == 0) {
						pred_box.score_ = scale;
					} else if (score_type == 1) {
						pred_box.score_ = max_prob;
					} else {
						pred_box.score_ = scale * max_prob;
					}
					int box_index = coord_index + k * 4 * locations;
					if (!constriant) {
						pred_box.box_.push_back(input_data[box_index + 0 * locations]);
						pred_box.box_.push_back(input_data[box_index + 1 * locations]);
					} else {
						pred_box.box_.push_back((i % side + input_data[box_index + 0 * locations]) / side);
						pred_box.box_.push_back((i / side + input_data[box_index + 1 * locations]) / side);
					}
					float w = input_data[box_index + 2 * locations];
					float h = input_data[box_index + 3 * locations];
					if (use_sqrt) {
						pred_box.box_.push_back(pow(w, 2));
						pred_box.box_.push_back(pow(h, 2));
					} else {
						pred_box.box_.push_back(w);
						pred_box.box_.push_back(h);
					}
					if (nms_threshold >= 0) {
						tmp_boxes.push_back(pred_box);
					} else {
						(*pred_boxes)[pred_label].push_back(pred_box);
					}
				}
			}
			if (nms_threshold >= 0)
			{
				std::sort(tmp_boxes.begin(), tmp_boxes.end(), [](const BoxData a, const BoxData b) {
					return a.score_ > b.score_;
				});
				vector<int> idxes;
				ApplyNms(tmp_boxes, &idxes, nms_threshold);
				for (int i = 0; i < idxes.size(); ++i) 
				{
					BoxData box_data = tmp_boxes[idxes[i]];
					if (pred_boxes->find(box_data.label_) == pred_boxes->end())
					{
						(*pred_boxes)[box_data.label_] = vector<BoxData>();
					}
					(*pred_boxes)[box_data.label_].push_back(box_data);
				}
			} else {
				for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it)
				{
					std::sort(it->second.begin(), it->second.end(), [](const BoxData a, const BoxData b) {
						return a.score_ > b.score_;
					});
				}
			}
		}
		
	private:
		FrameQueue *_in_queue;
		FrameQueue *_out_queue;

		int _side;
		int _num_objects;
		int _num_classes;
		bool _sqrt;
		bool _constriant;
		int _score_type;
		float _nms;
		float _score_threshold;
		string _label_map_name;
		std::map<int, string> _label_map;


		thread _thread;

		bool _pause_flag;
		IFrame* _iframe;
		cv::Mat _img;
};

class ActorDecorator: public ActorComponent {
	public:
		ActorDecorator() {
		}
		~ActorDecorator() {
			for (ActorComponent* _actor_component : _actor_component_vec)
				delete _actor_component;
		}
		/*virtual*/
		void operation() {
			for (ActorComponent* _actor_component : _actor_component_vec)
				_actor_component->operation();
		}
		/*virtual*/
		ActorDecorator* addcomponent(ActorComponent *actor_component) {
			_actor_component_vec.push_back(actor_component);
			return this;
		}
	private:
		vector<ActorComponent*> _actor_component_vec;
};

class HumanDetector: public IActor{
	public:
		HumanDetector(ACTOR_ARG &actor_arg, FrameQueue &in_queue, FrameQueue &out_queue) {
			_in_queue = &in_queue;
			_out_queue = &out_queue;

			_hitThreshold = 0;
			_winStride_x = 8;
			_winStride_y = 8;
			_padding_x = 32;
			_padding_y = 32;
			_scale = 1.05;
			_finalThreshold = 1.0;
		}
		~HumanDetector() {
			while (_thread.joinable())
				_thread.join();

			_img.release();
			assert( !_img.data);

        		// release resurce
			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			// Release YOLODetector
			cout << "del_HumanDetector" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "HumanDetector.init" << "\n";
			cout << "HumanDetector initialization Detector." <<"\n";	
			_hog = new HOGDescriptor();
			_hog->setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
			//namedWindow("people detector", 1);
			
		}
		/*virtual*/
		void operate() {
			cout << "HumanDetector.op " << "\n";
			string window_name = "test";
			while (1)
			{

				cout << "HumanDetector Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					cout << "HumanDetector  prepare image!" << "\n";
					_iframe = _in_queue->pop();

					_img = _iframe->img[0];

					vector<Rect> found, found_filtered;
					cout << "HumanDetector  detecting!" << "\n";
					double t = (double)getTickCount();
					_hog->detectMultiScale(_img, found, _hitThreshold, Size(_winStride_x, _winStride_y), Size(_padding_x, _padding_y), _scale, _finalThreshold);
					t = (double)getTickCount() - t;
					printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());

					size_t i, j;
					for( i = 0; i < found.size(); i++ )
					{
						Rect r = found[i];
						for( j = 0; j < found.size(); j++ )
							if( j != i && (r & found[j]) == r)
								break;
						if( j == found.size() )
							found_filtered.push_back(r);
					}
					for( i = 0; i < found_filtered.size(); i++ )
					{
						Rect r = found_filtered[i];
						// the HOG detector returns slightly larger rectangles than the real objects.
						// so we slightly shrink the rectangles to get a nicer output.
						r.x += cvRound(r.width*0.1);
						r.width = cvRound(r.width*0.8);
						r.y += cvRound(r.height*0.07);
						r.height = cvRound(r.height*0.8);

						BBOX_META *bbox_data = new BBOX_META{r.x, r.y, r.width, r.height, "people", 1.0};
						_iframe->pred_vec.push_back(bbox_data); 
					}
					_out_queue->push(_iframe);
					//cout << "HumanDetector Release iframe !" << "\n";
        				// release resurce
					//delete _iframe;
					//_iframe = nullptr;
					//assert( !_iframe);
				}
				else
					break;
			}
		}
		/*virtual*/
		void operation() {
			_thread = thread( &IActor::operate, this);
		}
		
	private:
		FrameQueue *_in_queue;
		FrameQueue *_out_queue;

		double _hitThreshold;
		int _winStride_x;
		int _winStride_y;
		int _padding_x;
		int _padding_y;
		double _scale;
		double _finalThreshold;
		HOGDescriptor *_hog;
		thread _thread;

		bool _pause_flag;
		IFrame* _iframe;
		cv::Mat _img;
};

class ActorDecorator_1st: public ActorDecorator {
	public:
		ActorDecorator_1st(): ActorDecorator(){
		}
		~ActorDecorator_1st() {
			cout << "del_ActorDecorator_1st" << "   \n";
		}
		/*virtual*/
		void operation() {
			ActorDecorator::operation();
			cout << "ActorDecorator_1st.op " << "\n";
		}
		/*virtual*/
		ActorDecorator* addcomponent(ActorComponent *actor_component) {

			cout << "ActorDecorator_1st.addcomponent " << "\n";
			return ActorDecorator::addcomponent(actor_component);
		}
};

class ActorFactory {
	public:
		virtual ~ActorFactory(){}
		virtual ActorComponent* createActorComponent(ACTOR_ARG &pactor_arg, FrameQueue& in_queue, FrameQueue& out_queue) = 0;
		virtual ActorComponent* createActorComponent(ACTOR_ARG &pactor_arg, FrameQueue& queue) = 0;
};

class ConcreteActorFactory: public ActorFactory{
	public:
		~ConcreteActorFactory(){
			cout << "\n" << "del_ConcreteActorFactory" << "\n";
		}

		ActorComponent* createActorComponent(ACTOR_ARG &actor_arg, FrameQueue &in_queue, FrameQueue &out_queue){

			cout << "create " << actor_arg.name << "\n"; 
			if (actor_arg.name == "gti_inferencer")
			{
				IActor *actor = new GTIInferencer(actor_arg, in_queue, out_queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
			if (actor_arg.name == "caffe_inferencer")
			{
				IActor *actor = new CaffeInferencer(actor_arg, in_queue, out_queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
			if (actor_arg.name == "yolo_detector")
			{
				IActor *actor = new YOLODetector(actor_arg, in_queue, out_queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
			if (actor_arg.name == "human_detector")
			{
				IActor *actor = new HumanDetector(actor_arg, in_queue, out_queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
		}

		ActorComponent* createActorComponent(ACTOR_ARG &actor_arg, FrameQueue &queue){

			cout << "create " << actor_arg.name << "\n";
			if (actor_arg.name == "video_reader")
			{
				IActor *actor = new VideoReader(actor_arg, queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			} 
			if (actor_arg.name == "image_reader")
			{
				IActor *actor = new ImageReader(actor_arg, queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			} 
			if (actor_arg.name == "image_shower")
			{
				IActor *actor = new ImageShower(actor_arg, queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
			if (actor_arg.name == "gti_fc_classifier")
			{
				IActor *actor = new GTIFcClassifier(actor_arg, queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}

		}

};
/*
typedef struct {
	// genernal
	char *name;
	// ImageShower
	char *disp_name;
	// VideoReader
	char *source;
	int width;
	int height;
	// GTIInferencer
	int gti_input_image_width;
	int gti_input_image_height;
	int gti_input_image_channel;
	int gti_device_type;  //(FTDI:0 EMMC:1 PCIE:2)
	char *gti_device_name;
	char *gti_nn_weight_name;
	char *gti_user_config;
	// GTIFcClassifier
	char *gti_fc_name;
	char *gti_fc_label;
	// CaffeInferencer
	char *caffe_model_file;
	char *caffe_trained_file;
	char *caffe_mean_file;
	char *caffe_label_file;
} ACTOR_ARG;
*/
int main() { 
	
	int gti_device_type = 0;
#ifdef USE_PCIE
	gti_device_type = 2;
#endif //USE_PCIE

	ACTOR_ARG actorARG[] =
	{
		{(char *)"video_reader", (char *)"", (char *)"./video_test/VideoDemoFastestMP4.mp4", 224, 224},
		{(char *)"gti_fc_classifier", (char *)"", (char *)"", 0, 0, 224, 224, 3, 0, (char *)"0", (char *)"", (char *)"", (char *)"/usr/local/GTISDK/data/Models/gti2801/gnet1/fc/picture_coef.bin", (char *)"/usr/local/GTISDK/data/Models/gti2801/gnet1/fc/picture_label.txt"},
		{(char *)"caffe_inferencer", (char *)"", (char *)"", 0, 0, 0, 0, 0, 0, (char *)"0", (char *)"", (char *)"", (char *)"", (char *)"", (char *)"/root/workspaces/vgg_yolo/gnet_deploy.prototxt", (char *)"/root/workspaces/vgg_yolo/gnet_yolo_iter_32000.caffemodel", (char *)"", (char *)"" },
		{(char *)"yolo_detector", (char *)"", (char *)"", 0, 0, 0, 0, 0, 0, (char *)"0", (char *)"", (char *)"", (char *)"", (char *)"", (char *)"", (char *)"", (char *)"", (char *)"" },
		{(char *)"image_shower", (char *)"disp3", (char *)"", 0, 0},
		{(char *)"human_detector", (char *)"", (char *)"", 0, 0}
	};

	ACTOR_ARG actorARG0[] =
	{
		{(char *)"video_reader", (char *)"", (char *)"./video_test/VideoDemoFastestMP4.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video1.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video2.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video3.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video4.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video5.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video6.mp4", 224, 224},
		{(char *)"video_reader", (char *)"", (char *)"./video_test/video7.mp4", 224, 224},
        };

	ACTOR_ARG actorARG1[] =
	{
		{(char *)"gti_inferencer", (char *)"", (char *)"", 0, 0, 224, 224, 3, gti_device_type, (char*)"0", (char*)"/root/workspaces/vgg_yolo/gnet1_coef_vgg16.dat", (char *)"/root/workspaces/vgg_yolo/userinput.txt"},
		{(char *)"gti_inferencer", (char *)"", (char *)"", 0, 0, 224, 224, 3, gti_device_type, (char*)"1", (char*)"/root/workspaces/vgg_yolo/gnet1_coef_vgg16.dat", (char *)"/root/workspaces/vgg_yolo/userinput.txt"},
		{(char *)"gti_inferencer", (char *)"", (char *)"", 0, 0, 224, 224, 3, gti_device_type, (char*)"2", (char*)"/root/workspaces/vgg_yolo/gnet1_coef_vgg16.dat", (char *)"/root/workspaces/vgg_yolo/userinput.txt"},
		{(char *)"gti_inferencer", (char *)"", (char *)"", 0, 0, 224, 224, 3, gti_device_type, (char*)"3", (char*)"/root/workspaces/vgg_yolo/gnet1_coef_vgg16.dat", (char *)"/root/workspaces/vgg_yolo/userinput.txt"},
	};


	FrameQueue  g1_queue;
	FrameQueue  g2_queue;
	FrameQueue  g3_queue;
	FrameQueue  g4_queue;

	FrameQueue  n_queue;
	FrameQueue  p_queue;
	FrameQueue  q_queue;


	ActorFactory *actor_factory = new ConcreteActorFactory();
	// video + cv human detector
	ActorComponent *slideshow = (new ActorDecorator_1st())->addcomponent(
			actor_factory->createActorComponent(actorARG[4], n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[5], g1_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[0], g1_queue)
			);

	/* 8 video + CHIP + CAFFE_YOLO
	ActorComponent *slideshow = (new ActorDecorator_1st())->addcomponent(
			actor_factory->createActorComponent(actorARG[4], q_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[3], p_queue, q_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[2], n_queue, p_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[2], n_queue, p_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[2], n_queue, p_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG1[3], g4_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG1[2], g3_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG1[1], g2_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG1[0], g1_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[7], g4_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[6], g3_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[5], g2_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[4], g1_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[3], g4_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[2], g3_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[1], g2_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG0[0], g1_queue)
			);
	*/

	/* pure video frame show
	ActorComponent *slideshow = (new ActorDecorator_1st())->addcomponent(
			actor_factory->createActorComponent(actorARG[5], m_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[0], m_queue)
			);
	*/


	slideshow->operation();
	cout << '\n';

	delete actor_factory;
	delete slideshow;

}
