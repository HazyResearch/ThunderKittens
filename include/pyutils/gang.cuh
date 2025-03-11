#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
    CUDA-specific ThreadPool

    Example usage

    // Construction
    KittensGang gang(device_ids, NUM_DEVICES);

    // Dispatch work to all threads (no need to set device)
    gang.execute([&](int dev_idx) {
        int dev;
        CUDACHECK(cudaGetDevice(&dev));
        if (dev != dev_idx) {
            fprintf(stderr, "Device mismatch: expected %d, got %d\n", dev_idx, dev);
            exit(1);
        }
    });
*/
class KittensGang {
public:
    KittensGang(const int *device_ids, const int num_devices);
    ~KittensGang();

    // Dispatches `task` to all threads, and waits for all threads to finish (using cv)
    void execute(std::function<void(int)> task);

private:
    // Condition indicators
    bool stop;
    std::vector<bool> task_available;
    int n_task_done;

    // Threadpool
    std::vector<std::thread> workers;
    
    // Main entry point for each thread
    void worker(int worker_id, int device_id);

    // Used to dispatch work to all threads
    std::function<void(int)> current_task;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};
    
KittensGang::KittensGang(const int *device_ids, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t i = 0; i < num_devices; ++i) {
        task_available.push_back(false);
        workers.emplace_back([this, i, device_ids] { worker(i, device_ids[i]); });
    }
}
    
KittensGang::~KittensGang() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}
    
void KittensGang::execute(std::function<void(int)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

void KittensGang::worker(int worker_id, int device_id) {
    cudaSetDevice(device_id); // done once and never again! This saves a LOT of time
    while (true) {
        std::function<void(int)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, worker_id] { return stop || task_available[worker_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[worker_id] = false;
        }
        task(worker_id);
        {
            std::lock_guard<std::mutex> lock(mutex); // adds about 10 microseconds overhead
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}
