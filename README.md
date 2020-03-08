### Training Workflow of Jumping Knowledge Network 
------

 - The very first thing you should do is put __GCN__ code library in the Main Branch __(Because of the size, it's not uploaded)__.
 - At first run the script __downloader.py__. It will automatically download necessary files in a folder named __Dataset__, that needed to train the network.  
 - Run the __main.py__ file for staring the training. A folder called __Visualization__ will be created, where you will find accuracy and loss graph plots.
 - Default training will run for __10__ folds, each fold having __300__ epochs.
 - For changing the fold size or No. of epochs, please refer to the __main.py__ file.