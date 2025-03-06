import threading
import subprocess
import time
import os

def run_batch_file(file):
    try:
        subprocess.run([file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def run_periodically(file, interval_seconds, max_time):
    start = time.time()
    do_terminate = False

    while not do_terminate:
        try:
            print(f"[Periodic Beamng Killer] Waiting {interval_seconds}s...")
            # Sleep for the specified interval (in seconds)
            time.sleep(interval_seconds)
            
            print("[Periodic Beamng Killer] Killing beamng periodically...")
            # Run the batch file
            run_batch_file(file)
        except Exception as e:
            print(e)
        except KeyboardInterrupt as k:
            print(k)
            do_terminate = True
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
    batch_thread.daemon = True
    
    # Start the thread
    batch_thread.start()

    return batch_thread

class Singleton(object):
        def __new__(cls, *args, **kwds):
            it = cls.__dict__.get("__it__")
            if it is not None:
                return it
            cls.__it__ = it = object.__new__(cls)
            it.init(*args, **kwds)
            return it

        def init(self, *args, **kwds):
            pass

class BeamngPeriodicKiller(Singleton):
    def init(self, *args, **kwds):
        self.kill_thread =  threading.Thread(target=run_periodically, 
                            args=(kwds["file"], 
                            kwds["interval_seconds"],
                            kwds["max_time"]))
        print("[BeamngPeriodicKiller] Thread created.")

    def start(self):
        if not self.kill_thread.is_alive():
            # Start the thread
            self.kill_thread.start()
            print("[BeamngPeriodicKiller] Thread started.")

    def get_thread(self):
        return self.kill_thread

if __name__ == "__main__":
    file = os.path.join(".", 'sims', 'beamng', 'kill_bmng.bat')

    killer = BeamngPeriodicKiller(file = file,
                                interval_seconds=5,
                                max_time = 10)
    killer.start()