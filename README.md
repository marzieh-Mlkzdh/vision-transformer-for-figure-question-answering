# vision-transformer-for-figure-question-answering
Figure Question Answering (FQA) is a multimodal task that tries to solve a high-level image understanding problem, where, for a scientific figure-question pair, e.g. does the red Gaussioan has the highest varaince?, the network should provide a correct answer. One of the main challenges in FQA is extracting rich features for figure represtantion. Furthermore, available dataset do not contain challenging figures. In this paper, we propose an FQA pipeline which incorporates the transformer architecture for enriched figure represtantion. This is in contrast to conventional methods that use CNNs. Additionally, we present the SBU-FQA dataset that is more diverse and challenging  compared to previous datasets. The dataset contains six different figure types with various figures in a single plot. Also, twenty-three question templates are created to make the task more challenging. The propsed method was evalauted on SBU-FQA, and showed significant improvements compared to the base-line and state of the art FQA methods. The results also indicated that the performance of previous method drops when tested on SBU-FQA dataset. 

# Model Structure 
![Group 188](https://user-images.githubusercontent.com/65096744/163026487-ba298f61-00e1-4051-bc72-b3d3fcc8fd1d.png)

# Download Dataset

To use our dataset (SBU-FQA) for training the provided python code. Easily use the link below for download.
https://figshare.com/articles/dataset/EFQA/17005654

# Train 
To train the model
1- firstly you need to download the dataset, the existing files are in json format and it is better to reform them into csv, some websites do it for free.
2- then run the python code image_feature_embedding.py and Question_Embedding.py and save the output files in your directory, 
3- finally you need to run the train.py . 

# Appendix
Here are some examples of model answers to a variety of questions.

![appe1](https://user-images.githubusercontent.com/65096744/163945684-4dbafdc0-846f-4fc0-8648-08e121aeefd4.JPG)
![appe2](https://user-images.githubusercontent.com/65096744/163945703-2ecbe246-0f2c-4f10-857c-5e451c295e2f.JPG)
![appe3](https://user-images.githubusercontent.com/65096744/163945731-8540a5d5-c307-4b01-a1ea-557aadfa7816.JPG)
![appe4](https://user-images.githubusercontent.com/65096744/163945751-d51b87c0-f93a-456b-874d-14a61e6267e1.JPG)
![appe5](https://user-images.githubusercontent.com/65096744/163945764-4e181f71-d271-40b4-8919-dc8130644de4.JPG)
![appe6](https://user-images.githubusercontent.com/65096744/163945798-2c8692eb-21b8-4b8f-9deb-7c5a3c2b51a3.JPG)
![appe7](https://user-images.githubusercontent.com/65096744/163945869-ff872e94-8c0e-462a-b05a-3f3fb38e2090.JPG)
