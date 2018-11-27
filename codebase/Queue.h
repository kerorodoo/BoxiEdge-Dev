//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Subject to the BSD 2-Clause License
// - see < http://opensource.org/licenses/BSD-2-Clause>
//

#ifndef CONCURRENT_QUEUE_
#define CONCURRENT_QUEUE_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>


/*****
 * Implement Lock-Free Queue
 *
 *****/
template <typename T>
class Queue
{
 public:
  
  /*****
   * get the first element in the Queue
   * 
   * return <T> element
   *
   *****/
  T pop() 
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    return val;
  }

  /*****
   * get the first element in the Queue
   * Parameter:
   *     item: set <T> element get from Queue
   *
   *****/
  void pop(T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
  }
  
  /*****
   * put the element into the Queue
   * Parameter:
   *     item: the <T> element put into Queue
   *
   *****/
  void push(const T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  size_t size()
  {  
    return queue_.size();
  }

  Queue()=default;
  Queue(const Queue&) = delete;            // disable copying
  Queue& operator=(const Queue&) = delete; // disable assignment
  
 private:
  std::queue<T> queue_;
  std::mutex mutex_;  //Lock the Queue for Thread-Safe
  std::condition_variable cond_;  //let thread blocked
};

#endif
