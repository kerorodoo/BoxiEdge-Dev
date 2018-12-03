/*
 * UML-diag Generate At https://yuml.me/diagram/scruffy/class/draw
 * Content As : [ActorComponent||+operation()]&lt;0..*-uses&lt;&gt;[ActorDecorator|-actor_component_vec|+operation();+addcomponent(){bg:orange}], [ActorComponent]^-.-[ActorDecorator], [ActorComponent]^-.-[IActor||+init(...);+operate();+operation()], [IActor]^-.-[Actor_etc..|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[IActor]^-.-[Actor_2nd|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()], [IActor]^-.-[VideoReader|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[ActorDecorator]^[ActorDecorator_1st], [ActorFactory||+createActorComponent(...){bg:green}]^[ConcreteActorFactory||+createActorComponent(...)], [ConcreteActorFactory]creates-.-&gt;[IActor],[user||main()]uses-&gt;[ActorFactory],[user||main()]uses-&gt;[ActorDecorator],[note: The ActorDecorator is an interface for the process such as Slideshow etc...]-.->[ActorDecorator],[note: The IActor is an interface for smallest unit in a process such as Camera  DataProcess NNInference etc...]-.->[IActor]
 * Coding Style Following: https://www.kernel.org/doc/html/v4.10/process/coding-style.html 
 * Coding Style Following: https://en.cppreference.com
 */

// Example program
#include <iostream>
#include <string>
#include <vector> 
#include <thread>

#include "Queue.h" 

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

// GTI headers
#include "GTILib.h"
#include "GtiClassify.h"
#include "Classify.hpp"

#define ImageQueue Queue<cv::Mat>
#define StreamQueue Queue<std::ostream>
#define DATA_PATH  "./";

using namespace std;
using namespace cv;

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
	// GTIFcClassify
	char *gti_fc_name;
	char *gti_fc_label;
} ACTOR_ARG;

typedef struct {
	cv::Rect rect;
	std::vector<std::string> label;
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
			
			cout << "IFrame Release buffer" << "\n";
			if (buffer != nullptr)
			{
				delete buffer;
				buffer = nullptr;
			}
		}
	public:
		std::vector<cv::Mat> img;
		std::vector<BBOX_META> rect_vec;
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
					iframe->img.push_back(cv::Mat(_img1));
					//iframe->img.push_back(_img1.clone());

					_out_queue->push(iframe);

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

class ImageShower: public IActor{
	public:
		ImageShower(ACTOR_ARG &actor_arg, FrameQueue &in_queue) {
			_in_queue = &in_queue;
			_window_name = actor_arg.disp_name;
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
				cout << "Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					_iframe = _in_queue->pop();

					cout << "ImageShower _img.deallocate" << "\n";

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
					//std::vector<cv::Mat> input_image;
					cv::Mat input_image;
					// converts input image to the float point 3 channel image.
					//cnnSvicProc32FC3(img, &input_image);
					cout << "GTIInferencer img CV_8U convertTo CV_32F !" << "\n";
					img.convertTo(input_image, CV_32F);
					// converts 32 bit float format image to 8 bit integer format image. 
					cout << "GTIInferencer img 32F convertTo 8Byte !" << "\n";
					cvt32FloatTo8Byte((float *)input_image.data, (uchar *)_input_image_buffer, _gti_input_image_width, _gti_input_image_height, _gti_input_image_channel);

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
					input_image.release();

					iframe->buffer = _nn_output_buffer;
					
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
		unsigned char* _input_image_buffer; 

		bool _pause_flag;
};

class GTIFcClassify: public IActor{
	public:
		GTIFcClassify(ACTOR_ARG &actor_arg, FrameQueue &in_queue) {
			_in_queue = &in_queue;

			_gti_fc_name = actor_arg.gti_fc_name;
			_gti_fc_label = actor_arg.gti_fc_label;
		}
		~GTIFcClassify() {
			while (_thread.joinable())
				_thread.join();
        		// release resurce
			delete _iframe;
			_iframe = nullptr;
			assert( !_iframe);
			// Release FC
			GtiClassifyRelease(_gti_classify);
			cout << "del_GTIFcClassify" << '\n';
		}
		/*virtual*/
		void init() {
			cout << "GTIFcClassify.init" << "\n";
			cout << "GTIFcClassify initialization FC." <<"\n";
			_gti_classify = GtiClassifyCreate(_gti_fc_name.c_str(), _gti_fc_label.c_str());
						
		}
		/*virtual*/
		void operate() {
			cout << "GTIFcClassify.op " << "\n";
			while (1)
			{
				cout << "GTIInferencer Queue.size:" << _in_queue->size() << " ... \n";
				if (_pause_flag == 0 || _in_queue->size() != 0)
				{
					_iframe = _in_queue->pop();
					float *_nn_output_buffer = _iframe->buffer;
					
					cout << "GTIFcClassify GtiClassifyFC !" << "\n";
					GtiClassifyFC(_gti_classify, _nn_output_buffer, 5);

					/* Print the top N predictions. */	
					for (int i = 0; i < 5; ++i)
					{
						char *ptext = GetPredicationString(_gti_classify, i);
						cout << "GTIFcClassify ptext: " << ptext << "\n";
					}

					cout << "GTIFcClassify Release iframe !" << "\n";

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

		Classify *_gti_classify;

		bool _pause_flag;
		IFrame *_iframe;
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
			if (actor_arg.name == "image_shower")
			{
				IActor *actor = new ImageShower(actor_arg, queue);
				cout << "init " << actor_arg.name << "\n"; 
				actor->init();
				return actor;
			}
			if (actor_arg.name == "gti_fc_classify")
			{
				IActor *actor = new GTIFcClassify(actor_arg, queue);
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
	// GTIFc
	char *gti_fc_name;
	char *gti_fc_label;
} ACTOR_ARG;
*/
int main() { 
	ACTOR_ARG actorARG[] =
	{
		{(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
		{(char *)"gti_inferencer", (char *)"", (char *)"", 0, 0, 224, 224, 3, 0, (char*)"0", (char*)"/usr/local/GTISDK/data/Models/gti2801/gnet1/cnn/gnet1_coef_vgg16.dat", (char *)"/usr/local/GTISDK/data/Models/gti2801/gnet1/cnn/userinput.txt"},
		{(char *)"gti_fc_classify", (char *)"", (char *)"", 0, 0, 224, 224, 3, 0, (char *)"0", (char *)"", (char *)"", (char *)"/usr/local/GTISDK/data/Models/gti2801/gnet1/fc/picture_coef.bin", (char *)"/usr/local/GTISDK/data/Models/gti2801/gnet1/fc/picture_label.txt"},
		{(char *)"image_shower", (char *)"disp3", (char *)"", 0, 0}
	};
	FrameQueue  m_queue;
	FrameQueue  n_queue;
	FrameQueue  p_queue;


	ActorFactory *actor_factory = new ConcreteActorFactory();
	ActorComponent *slideshow = (new ActorDecorator_1st())->addcomponent(
			actor_factory->createActorComponent(actorARG[2], n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[2], n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[1], m_queue, n_queue))->addcomponent(
			actor_factory->createActorComponent(actorARG[0], m_queue)
			);


	slideshow->operation();
	cout << '\n';

	delete actor_factory;
	delete slideshow;

}
