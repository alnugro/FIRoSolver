import multiprocessing
import time

def worker(queue):
    for i in range(5):
        time.sleep(1)
        queue.put(f"Task {i} done")

if __name__ == '__main__':
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(queue,))
    p.start()

    # Main process listening for messages (similar to receiving signals)
    while p.is_alive() or not queue.empty():
        try:
            message = queue.get(timeout=1)  # Timeout prevents hanging indefinitely
            print(f"Received: {message}")
        except multiprocessing.queues.Empty:
            continue

    p.join()
