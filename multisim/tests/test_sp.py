import threading
import subprocess
import time
import os
import asyncio

def run_batch_file(file):
    try:
        subprocess.run([file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        
async def wait_time(time):
    await asyncio.sleep(time)

def run_periodically(file, interval_seconds, max_time):
    start = time.time()
    do_terminate = False

    while not do_terminate:
        try:
            print(f"[Periodic Beamng Killer] Waiting {interval_seconds}s...")
            for _ in range(interval_seconds*1000):
                # Sleep for the specified interval (in seconds)
                try:
                    # await asyncio.sleep(1/10)
                    # asyncio.create_task(wait_time(1/10))

                    print("sleeping")
                except Exception as e:
                    sys.exit(0)
            print("[Periodic Beamng Killer] Killing beamng periodically...")
            # Run the batch file
            run_batch_file(file)
        except KeyboardInterrupt as k:
            print("[PBK] Keyboard interrupt ocurred.")
            print(k)
            do_terminate = True
            sys.exit(0)
        # except Exception as e:
        #     print("[PBK] Exception ocurred.")
        #     print(e)
        finally:
            if (time.time() - start) > max_time:
                print("[Periodic Beamng Killer] Maximal time reached. Terminating.")
                do_terminate = True


def kill_periodically_beamng(interval_seconds = 600, 
                            file = "/sims/beamng/kill_bmng.bat",
                            max_time = 86400 # one day
                            ):
    # Create a thread to run the batch file periodically
    batch_thread = threading.Thread(target=run_periodically, 
                            args=(file, interval_seconds, max_time))
    batch_thread.daemon = False
    
    # Start the thread
    batch_thread.start()

    return batch_thread

if __name__ == "__main__":
    # main loop
    i = 10
    kill_periodically_beamng(interval_seconds = 2, 
                            file = r"echo.bat",
                            max_time = 20 # one day
                            )
    for i in range(0,i):
        print(f"main loop: {i}")

