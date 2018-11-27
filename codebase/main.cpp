//uml-diag
/*<img src="http://y...content-available-to-author-only...l.mediagram/scruffy/class/[ActorComponent||+operation()]&lt;0..*-uses&lt;&gt;[ActorDecorator|-actor_component_vec|+operation();+addcomponent(){bg:orange}], [ActorComponent]^-.-[ActorDecorator], [ActorComponent]^-.-[IActor||+init(...);+operate();+operation()], [IActor]^-.-[Actor_etc..|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[IActor]^-.-[Actor_2nd|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()], [IActor]^-.-[VideoReader|-in_queue;-out_queue;-thread|+init(...);+operate();+operation()],[ActorDecorator]^[ActorDecorator_1st], [ActorFactory||+createActorComponent(...){bg:green}]^[ConcreteActorFactory||+createActorComponent(...)], [ConcreteActorFactory]creates-.-&gt;[IActor],[user||main()]uses-&gt;[ActorFactory],[user||main()]uses-&gt;[ActorDecorator],[note: The ActorDecorator is an interface for the process such as Slideshow etc...]-.->[ActorDecorator],[note: The IActor is an interface for smallest unit in a process such as Camera  DataProcess NNInference etc...]-.->[IActor]">*/
 
// Example program
#include <iostream>
#include <string>
#include <vector> 
#include <thread>

#include "Queue.h" 

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#define ImageQueue Queue<cv::Mat>
#define StreamQueue Queue<std::ostream>
#define DATA_PATH  "./";
 
using namespace std;
using namespace cv;

typedef struct {
    char *name;
    char *disp_name;
    char *source;
    int width;
    int height;
} ACTOR_ARG;
 
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
    VideoReader(ACTOR_ARG &actor_arg, ImageQueue& out_queue) {
    	_out_queue = &out_queue;
        _demofile = actor_arg.source;
        _width1 = actor_arg.width;
        _height1 = actor_arg.height;
    }
    ~VideoReader() {
    	while(_thread.joinable())
    	    _thread.join();
        cout << "del_VideoReader" << '\n';
    }
    /*virtual*/
    void init() {
    	cout << "VideoReader.init" << "\n";


        std::cout << "Open Video file " << _demofile << " ...\n";
        _cap.open(_demofile);

        if (!_cap.isOpened())
        {
            _cap.release();
            std::cerr << "Video Clip File " << _demofile << " is not opened!\n";
            return;
        }

        _width  = _cap.get(CV_CAP_PROP_FRAME_WIDTH);
        _height  = _cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        _frame_count  = _cap.get(CV_CAP_PROP_FRAME_COUNT);
        int video_cc = _cap.get(CV_CAP_PROP_FOURCC);
        _video_codec  = format("%c%c%c%c", video_cc & 255, (video_cc >> 8 ) & 255, (video_cc >> 16 ) & 255, (video_cc >> 24 ) & 255 );
        _pause_flag = 0;
        
        _frame_num = 0;
    }
    /*virtual*/
    void operate() {
        cout << "VideoReader.op " << "\n";
        while(1)
        {
            if (_pause_flag == 0)
            {
                bool cap_read_false;
                // reset if video playing ended
                if (_frame_num == _frame_count)
                {
                    _cap.set(CV_CAP_PROP_POS_FRAMES, 0);
                    _frame_num = 0;
                }
                _cap.read(_img);
                _frame_num++;
                

                std::cout << "_video_codec: " << _video_codec <<  "  ...\n";
                std::cout << "_frame_num: " << _frame_num << "/" << _frame_count << "  ...\n";
                cv::resize(_img, _img1, cv::Size(_width1, _height1));
                _out_queue->push(_img);
            }
            else
                break;
        }
    }
 
    /*virtual*/
    void operation() {
        _thread = thread(&IActor::operate, this);
    }
 
  private:
    ImageQueue* _out_queue;
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
 
class Actor_2rd: public IActor{
  public:
    Actor_2rd(ACTOR_ARG &actor_arg, ImageQueue& in_queue, ImageQueue& out_queue) {
    	_in_queue = &in_queue;
    	_out_queue = &out_queue;
        _window_name = actor_arg.disp_name;
    }
    ~Actor_2rd() {
    	while(_thread.joinable())
    	    _thread.join();
        cout << "del_Actor_2nd" << '\n';
    }
    /*virtual*/
    void init() {
    	cout << "Actor_2nd.init" << "\n";
    }
    /*virtual*/
    void operate() {
        cout << "Actor_2nd.op " << "\n";
        while(1)
        {
            cout << "Queue.size:" << _in_queue->size() << " ... \n";
            if (_pause_flag == 0 || _in_queue->size() != 0)
            {
                _img = _in_queue->pop();
                //cv::imshow(_window_name, _img);
                cv::waitKey(1);
                _img.release();
            }
            else
                break;
        }
    }
    /*virtual*/
    void operation() {
        _thread = thread(&IActor::operate, this);
    }
  private:
    ImageQueue* _in_queue;
    ImageQueue* _out_queue;
    string _window_name;
    thread _thread;
 
    bool _pause_flag;
    cv::Mat _img;
};
 
class ActorDecorator: public ActorComponent {
  public:
    ActorDecorator() {
    }
    ~ActorDecorator() {
    	for (ActorComponent* _actor_component: _actor_component_vec)
            delete _actor_component;
    }
    /*virtual*/
    void operation() {
    	for (ActorComponent* _actor_component: _actor_component_vec)
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
    virtual ActorComponent* createActorComponent(ACTOR_ARG &pactor_arg, ImageQueue& in_queue, ImageQueue& out_queue) = 0;
    virtual ActorComponent* createActorComponent(ACTOR_ARG &pactor_arg, ImageQueue& queue) = 0;
};
 
class ConcreteActorFactory: public ActorFactory{
  public:
    ~ConcreteActorFactory(){
    	cout << "\n" << "del_ConcreteActorFactory" << "\n";
    }

    ActorComponent* createActorComponent(ACTOR_ARG &actor_arg, ImageQueue& in_queue, ImageQueue& out_queue){
 
    	cout << "create " << actor_arg.name << "\n"; 
    	if (actor_arg.name == "gti_inference")
    	{
    	    IActor *actor = new Actor_2rd(actor_arg, in_queue, out_queue);
    	    actor->init();
    	    return actor;
    	}
    }

    ActorComponent* createActorComponent(ACTOR_ARG &actor_arg, ImageQueue& queue){
 
    	cout << "create " << actor_arg.name << "\n";
    	if (actor_arg.name == "video_reader")
    	{
    	    IActor *actor = new VideoReader(actor_arg, queue);
    	    actor->init();
    		return actor;
    	} 
 
    }
 
};
 
int main() { 
    ACTOR_ARG actorARG[] =
    {
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"video_reader", (char *)"", (char *)"./VideoDemoFastestMP4.mp4", 224, 224},
        {(char *)"gti_inference", (char *)"disp3", (char *)"", 0, 0}
    };
    ImageQueue m_queue;
    ImageQueue n_queue;
   
    ActorFactory *actor_factory = new ConcreteActorFactory();
    ActorComponent *slideshow = (new ActorDecorator_1st())->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[8], m_queue, n_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[7], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[6], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[5], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[4], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[3], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[2], m_queue))->addcomponent(
    	                     actor_factory->createActorComponent(actorARG[1], m_queue))->addcomponent(
                               actor_factory->createActorComponent(actorARG[0], m_queue)
                               );
  
   
    slideshow->operation();
    cout << '\n';
   
    delete actor_factory;
    delete slideshow;
   
}
