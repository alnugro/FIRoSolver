from pebble import ProcessPool, ProcessExpired
import time

def count_to(limit):
    count = 0
    while count < limit:
        count += 1
    return limit

def main():
    with ProcessPool(max_workers=4) as pool:
        future1 = pool.schedule(count_to, args=(1000000000000,), timeout=10)
        future2 = pool.schedule(count_to, args=(10000000000,), timeout=10)
        future3 = pool.schedule(count_to, args=(1000000000,), timeout=10)
        futures = [future1, future2, future3]

        try:
            # Wait for any future to complete
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    print(f"Process completed: Counted to {result}")

                    # If any process finishes, cancel the others
                    for f in futures:
                        if f is not future and not f.done():
                            f.cancel()
                            print("Cancelled another process")

                    break

                except TimeoutError:
                    print("Process took too long (TimeoutError)")

                except ProcessExpired as error:
                    print(f"Process {error.pid} took too long (ProcessExpired)")

                except Exception as error:
                    print(f"Process raised {error}")

        finally:
            # Ensure all futures are cancelled if not already done
            for future in futures:
                if not future.done():
                    future.cancel()

if __name__ == "__main__":
    main()
