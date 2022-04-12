# vision-transformer-for-figure-question-answering
Figure Question Answering (FQA) is a multimodal task that tries to solve a high-level image understanding problem, where, for a scientific figure-question pair, e.g. does the red Gaussioan has the highest varaince?, the network should provide a correct answer. One of the main challenges in FQA is extracting rich features for figure represtantion. Furthermore, available dataset do not contain challenging figures. In this paper, we propose an FQA pipeline which incorporates the transformer architecture for enriched figure represtantion. This is in contrast to conventional methods that use CNNs. Additionally, we present the SBU-FQA dataset that is more diverse and challenging  compared to previous datasets. The dataset contains six different figure types with various figures in a single plot. Also, twenty-three question templates are created to make the task more challenging. The propsed method was evalauted on SBU-FQA, and showed significant improvements compared to the base-line and state of the art FQA methods. The results also indicated that the performance of previous method drops when tested on SBU-FQA dataset. 

# Model Structure 
![Group 188](https://user-images.githubusercontent.com/65096744/163026487-ba298f61-00e1-4051-bc72-b3d3fcc8fd1d.png)

# Download Dataset

To use our dataset (SBU-FQA) for training the provided python code. Easily use the link below for download.
https://figshare.com/articles/dataset/EFQA/17005654

# Sample data
![Figure7](https://user-images.githubusercontent.com/65096744/163022734-e417ed48-15ea-4a57-b2ce-6eea08f731f3.png)
![Figure5](https://user-images.githubusercontent.com/65096744/163022804-a255cf36-f7c1-46d2-9ef6-31af26a75a61.png)
![S8_Fig](https://user-images.githubusercontent.com/65096744/163022894-da5d8456-5246-4de9-9f76-a2d1cdb7839e.png)
![S9_Fig](https://user-images.githubusercontent.com/65096744/163022901-0daccfb1-f9a2-4b3e-bc57-f417bcd71ccf.png)
![S14_Fig](https://user-images.githubusercontent.com/65096744/163022910-3cc5de10-85aa-4569-ba5c-c72c2034e7d4.png)
![S17_Fig](https://user-images.githubusercontent.com/65096744/163022918-54f3b83c-89d0-4dd7-b33e-7a90b41d5a72.png)
![S18_Fig](https://user-images.githubusercontent.com/65096744/163022934-144ac332-bcee-44bb-be2d-52606b81dd27.png)
![S10_Fig](https://user-images.githubusercontent.com/65096744/163022953-14dec422-1358-467c-b03b-f8162754de6b.png)
![S19_Fig](https://user-images.githubusercontent.com/65096744/163022960-b21364a5-7b60-4409-a098-a7581172eccd.png)
![Figure4](https://user-images.githubusercontent.com/65096744/163022989-93faebe5-86c5-4e2e-92b4-40a503d15b4c.png)

# Train 
To train the model
1- firstly you need to download the dataset, the existing files are in json format and it is better to reform them into csv, some websites do it for free.
2- then run the python code image_feature_embedding.py and Question_Embedding.py and save the output files in your directory, 
3- finally you need to run the train.py . 

# Appendix

