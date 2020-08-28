
class DataLoader:
    ''' The Sidekick Dataloader Class \n
    Handles multithreaded data loading, batching, preparing, etc... \n
    Inputs:
        batch_size (int): The size of batches to feed out
        load_function (function) Parameters: (loader (the loader class), index (the index to be loaded)): The function responsible for directly loading examples.  
            This function should load a single example if the collate_function is defined, or a batch of examples if not. 
            What gets returned will either be fed directly to the collate function, or directly out of the iterator.
        init_function (function) Parameters: (loader (the loader class), start_index (passed into the dataloader), end_index (passed into the dataloader)) [Default: None]: The function for handling initialization. Should store all nessacary variables in the loader class as attributes.
        collate_function (function) Parameters: (loader (the loader class), batch (the list of batch_size outputs from the load_function)) [Default: None]: The function responsible for combining batch_size examples into a batch. Any additional preprocessing should be put in here. 
            If this function is not specified, the output of one call of the load_function will be assumed to be a full batch and be returned from the iterator.
        start_index (int) [Default: 0]: The index to start loading the dataset from. This index will be the lowest one passed into the load_function. This will also be fed into the init_function.
        end_index (int) [Default: None]: The index to stop loading the dataset. This index will be the highest one passed into the load_function. This will also be fed into the init_function.
    '''
    def __init__(self, batch_size, load_function, init_function=None, collate_function=None, start_index=0, end_index=None, preload=False, num_workers=0):
        self.batch_size = batch_size
        self.load_function = load_function
        self.collate_function = collate_function
        
        if init_function is not None: init_function(self, start_index, end_index)

        self.start_index = start_index
        self.end_index = end_index

        self.batch_queue = JoinableQueue()

        if preload:
            # Call the loading function and join all of the created processes before returning fron the __init__ function
            self.load_data(start_index, end_index)
            # Join the processes
            for process in multiprocessing.active_children():
                process.join()
            # Make preloaded list
            self.preload_list = []
            while not self.batch_queue.empty():
                self.preload_list.append(self.batch_size.get())
        
        
    def __iter__(self):
        return iter(self.preload_list) if self.preload else self

    def __next__(self):
        

    def load_data(self, start_index, end_index):
        # Divide the data into num_workers slices to feed into each worker
        slice_size = (end_index - start_index) // self.num_workers
        for i in range(self.num_workers):
            job = Process(target=load_job, args=(self, start_index + (slice_size * i), start_index + (slice_size * (i + 1))))
            job.start()


def load_job(loader, start_index, end_index): # The job to be run in each worker
    for i in range(end_index - start_index):
        self.batch_queue.put(self.load_function(self, i))