import subprocess, time, threading

class SyncingDirectory:
    def __init__(self,source,destination,sync_deletions=False,sync_time=60):
        self.source = source
        self.destination = destination
        self.stop = threading.Event()
        self.syncing_thread = threading.Thread(target=self._sync,args=(),daemon=True)
        self.sync_deletions = sync_deletions
        self.sync_time = sync_time

    def _sync(self):
        if self.sync_deletions:
            command = ['rsync','-aP',f'{self.source}/',f'{self.destination}','--delete']
        else: 
            command = ['rsync','-aP',f'{self.source}/',f'{self.destination}']
        while not self.stop.is_set():
            subprocess.run(command)
            time.sleep(self.sync_time)

    def quick_sync(self):
        if self.sync_deletions:
            command = ['rsync','-aP',f'{self.source}/',f'{self.destination}','--delete']
        else: 
            command = ['rsync','-aP',f'{self.source}/',f'{self.destination}']
        subprocess.run(command)
        return True
    
    def background_sync(self,verbose=False):
        if self.syncing_thread.is_alive():
            if verbose: print("Active thread detected... ",end="")
            self.stop.set()
            self.syncing_thread.join()
            if verbose: print("Stopped.")
        if self.syncing_thread._started.is_set():
            if verbose: print("Creating a fresh new thread... ",end="")
            self.syncing_thread = threading.Thread(target=self._sync,args=(),daemon=True)
            if verbose: print("Done.")
        if self.stop.is_set():
            if verbose: print("Creating new stop event... ",end="")
            self.stop.clear()
            if verbose: print("Done.")
        if verbose: print("Starting new thread...",end="")
        self.syncing_thread.start()
        if verbose: print("Done!")
        return True

    def print_status(self):
        print(f"The background thread has been started: {self.syncing_thread._started.is_set()}")
        print(f"The background thread is alive: {self.syncing_thread.is_alive()}")
        print(f"The background thread is stopped: {self.stop.is_set()}")
