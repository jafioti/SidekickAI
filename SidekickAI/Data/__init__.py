import multiprocessing
import time, torch, math, random
from SidekickAI.Data import batching

class Dataset:
    ''' The Sidekick Dataloader Class \n
    Handles multithreaded data loading, batching, preparing, etc... \n
    Inputs:
        batch_size (int): The size of batches to feed out
        load_function (function) Parameters: (data (a dict with only the current index of data in it), other (a dict with other random things in it) global_index (the index to be loaded, global based on the start/end indexes passed into the dataset)): The function responsible for directly loading examples.  
            This function should load a single example if the collate_function is defined, or a batch of examples if not. 
            What gets returned will either be fed directly to the collate function, or directly out of the iterator.
        end_index (int): The index to stop loading the dataset. This index will be the highest one passed into the load_function. This will also be fed into the init_function.
        init_function (function) Parameters: (loader (the loader class), start_index (passed into the dataset), end_index (passed into the dataset)) [Default: None]: The function for handling initialization. Should store all nessacary variables in the data dict and the other dict.
        collate_function (function) Parameters: (batch (the list of batch_size outputs from the load_function), other (dict containing non data stuff)) [Default: None]: The function responsible for combining batch_size examples into a batch. Any additional preprocessing should be put in here. 
            If this function is not specified, the output of one call of the load_function will be assumed to be a full batch and be returned from the iterator.
        start_index (int) [Default: 0]: The index to start loading the dataset from. This index will be the lowest one passed into the load_function. This will also be fed into the init_function.
        preload (bool) [Default: False]: Whether or not to load all the data on initialization or not.
        num_workers (int) [Default: 0]: The number of multiprocessing processes to use to load data. The load/collate functions are called from these processes. If num_workers = 0, all loading will be done syncronously.
        data_chunk_size (int) [Default: None (Dynamic)]: The max number of examples to be loaded in at once. If left as none, will be decided dynamically based on full dataset size. Practically this will never be hit, but it is the theoretical max.
    '''
    def __init__(self, batch_size, load_function, end_index, init_function=None, collate_function=None, start_index=0, preload=False, num_workers=0, data_chunk_size=None, **kwargs):
        self.__dict__.update(kwargs) # For any custom variables the user wants to pass in
        self.batch_size = batch_size
        self.preload = preload
        self.num_workers = num_workers if not preload else 0 # Haven't figured out a good way to preload with multiprocessing yet, so defaults to sync loading
        self.start_index, self.end_index = start_index, end_index
        self.load_function = load_function
        self.collate_function = collate_function
        self.data, self.other = {}, {}
        
        if init_function is not None: init_function(self, start_index, end_index)
        for (key, value) in self.data.items(): assert isinstance(value, list), str(key) + " is not a list. All items in the data dict must be a list!" # Make sure all items in the data dict are lists
        assert len(set([len(value) for (key, value) in self.data.items()])) <= 1, "Not all lists are of the same length!" # Ensure all of the lists are of the same length
        self.end_index = start_index + len(self.data[list(self.data.keys())[0]]) # Ensure the end_index is not furthur than the data itself
        self.batch_queue = multiprocessing.JoinableQueue() if not preload else []

        if preload:
            # Call the loading function and join all of the created processes before returning fron the __init__ function
            self.load_data(self.start_index, self.end_index)
            # Join the processes
            for process in multiprocessing.active_children(): process.join()

        self.loaded_index = start_index
        self.waits = 0
        self.iterations = 0
        # Find dynamic data_chunk_size
        self.data_chunk_size = min(min(max(int((self.end_index - self.start_index) / 5), 2000), 20000), self.end_index - self.start_index) if data_chunk_size is None else data_chunk_size # 2000: min, 20000: max | These are arbitrary

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]]) // self.batch_size if not self.preload else len(self.batch_queue)

    def example_len(self):
        return int(self.__len__() * self.batch_size)

    def __iter__(self):
        self.iterations = 0
        return iter(self.batch_queue) if self.preload else self

    def __next__(self):
        self.iterations += 1
        if self.iterations >= self.__len__(): # Stop iterating
            self.stop_iteration()

        while self.batch_queue.empty():
            self.waits += 1
            time.sleep(1)
            if self.waits > 2:
                for job in multiprocessing.active_children(): job.terminate()
                self.waits = 0
            if len(multiprocessing.active_children()) < self.num_workers or self.num_workers == 0:
                if self.loaded_index >= self.end_index - self.batch_size: self.stop_iteration()
                self.load_data(self.loaded_index, min(self.loaded_index + self.data_chunk_size, self.end_index - 1))
                self.loaded_index = min(self.loaded_index + self.data_chunk_size, self.end_index)
                self.waits = 0
                time.sleep(2)
        try:
            return self.batch_queue.get()
        except: return None

    def stop_iteration(self):
        self.loaded_index = self.start_index
        self.iterations = 0
        self.waits = 0
        self.batch_queue = multiprocessing.JoinableQueue()
        raise StopIteration

    def shuffle(self):
        # Shuffle data
        if self.preload:
            random.shuffle(self.batch_queue)
        else:
            lists = batching.shuffle_lists_retain_batches(self.batch_size, *self.data.values())
            for i, key in enumerate(self.data.keys()):
                self.data[key] = lists[i]

    def load_data(self, start_index, end_index):
        # If num_workers = 0, run load_job syncrounously
        if self.num_workers == 0:
            load_job({key:value[start_index - self.start_index:end_index - self.start_index] for (key, value) in self.data.items()}, self.other, start_index, end_index, self.batch_size, self.load_function, self.collate_function, self.batch_queue)
        else:
            # Divide the data into num_workers slices to feed into each worker
            slice_size = (end_index - start_index) // self.num_workers
            for i in range(self.num_workers):
                job = multiprocessing.Process(target=load_job, args=({key:value[start_index + (slice_size * i) - self.start_index:start_index + (slice_size * (i + 1)) - self.start_index] for (key, value) in self.data.items()}, self.other, start_index + (slice_size * i), start_index + (slice_size * (i + 1)), self.batch_size, self.load_function, self.collate_function, self.batch_queue))
                job.start()

def load_job(data, other, start_index, end_index, batch_size, load_function, collate_function, batch_queue): # The job to be run in each worker
    end_index = start_index + len(data[list(data.keys())[0]])
    for i in range(0, end_index - start_index - batch_size, batch_size):
        if collate_function is None:
            batch = load_function({key:value[i] for (key, value) in data.items()}, other, start_index + i) # Pass in data, other, global index
        else:
            batch = collate_function([load_function({key:value[x] for (key, value) in data.items()}, other, start_index + x) for x in range(i, min(i + batch_size, end_index - start_index))], other)
        # for item in batch: # Ensure all tensors are in shared memory to help with multiprocessing (NOT SURE IF THIS IS NEEDED, SOMETIMES CAUSES PROBLEMS)
        #     if isinstance(item, torch.Tensor):
        #         item.share_memory_()
        if isinstance(batch_queue, list): batch_queue.append(batch)
        else: batch_queue.put_nowait(batch)
    if not isinstance(batch_queue, list): batch_queue.join()