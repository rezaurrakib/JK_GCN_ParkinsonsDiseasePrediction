### Jumping Knowledge Network for Disease Prediction
Representation learning on Graphs using Jumping Knowledge (JK) networks published on ICML 2018 [(paper_link)](https://arxiv.org/pdf/1806.03536.pdf). In this research work, we tried to explore the concept of JK network on medical non-imaging dataset for parkinsons disease prediction.

The goal of this practical was two fold.
 - Disease prediction (2 class classification)
 - Exploit rich multi-modal data

### Parkinson's Disease (PD) and Dataset collection
 - PD is a Long-term degenerative disorder of the central nervous system.
 - Symptom includes shaking, rigidity, slow movement, difficulty with walking etc.
 
 We had used 4 different types of non-imaging data for our prediction task.
 - [MOCA (Montreal Cognitive Assessment)](https://www.mocatest.org/)
 - [UPDRS (Unified Parkinson’s Disease Rating Scale)](https://www.parkinsons.org.uk/professionals/resources/unified-parkinsons-disease-rating-scale-updrs)
 - Age
 - Gender
 
 Dataset Preprocessing
 -------
 #### Dataset
 - [PPMI](https://www.ncbi.nlm.nih.gov/pubmed/21930184) dataset (Parkinson's Progression Markers Initiative)
 - MRI imaging for feature extraction
 - Non-imaging data for affinity graph construction

#### Feature Extraction
 - Apply Autoencoder, get the bottleneck, flatten for every node/patient
 - LLE for dimensionality reduction
 - Embedding of 324 patients, 300 features for each

#### Affinity Graph Construction
 - Using screening assessment results. i.e., MOCA & UPDRS test.
 - Phenotype measures, i.e., Age & Gender

### Training Workflow of Jumping Knowledge Network 
------

 - At first put __GCN__ code library in the Main Branch. Download it from [GCN Github](https://github.com/tkipf/gcn).
 - Run the script __downloader.py__. It will automatically download necessary files in a folder named __Dataset__, that needed to train the network.  
 - Then run the __main.py__ file for staring the training. A folder called __Visualization__ will be created, where you will find accuracy and loss graph plots.
 - Default training will run for __10__ folds, each fold having __300__ epochs.
 - For changing the fold size or No. of epochs, please refer to the __main.py__ file.
