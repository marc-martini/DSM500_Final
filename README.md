# Developing Transformers for Electrical Power Demand Forecasting
# DSM 500 Reseach project 


Marc Martinho

The following is the full code base used to conduct the rearch for the Data Science DSM 500 - Final Research Project 
Title - Developing Transformers for Electrical Power Demand Forecasting


## Abstract
Accurately forecasting electrical power loads is crucial for the stability and reliability of modern power grids, especially as they integrate more renewable energy sources. This research explores the application of advanced transformer-based models with different data augmentation and pre-processing techniques, such as PatchTST and FEDFormer, in forecasting electrical load data. Several novel approaches are designed to investigate how these techniques affect the forecasting performance by comparing the performance of these models against current state-of-the-art baselines. The results demonstrate that PatchTST, which employs advanced data augmentation techniques, consistently achieves superior forecasting performance across various scenarios. However, findings also suggest that simpler may be more effective for shorter prediction horizons, emphasising the importance of model selection based on the specific application needs. Additionally, the impact of Reversible Instance (REVIn) normalisation was evaluated, showing enhanced performance in some cases but with varying effectiveness depending on the model architecture. The results highlight the potential of transformers to capture complex temporal dependencies in electrical data. Additionally, the research indicated the need to balance between model complexity and performance.
The research contributes to the field by providing a deeper understanding of transformer models and insights into model optimisation for time series forecasting. There are two main suggested areas for further exploration; reinforcement learning approaches and the application of data augmentation to other model architectures.


## Main Results
![image](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)

###  Keywords
|  Time series analysis:  |  The analysis of sequential data points, usually collected over time, to identify patterns, trends, and relationships in the data.  | 
|:--:|:--:|
|  Time series forecasting:  |  The forecasting of future data points based on previously observed time series data, using statistical models or machine learning techniques.  |
|  Electrical Data forecasting:  |  The forecasting of future electrical consumption, generation, or load based on historical electrical data and patterns.  |
|  Transformers:  |  A type of deep learning architecture designed for handling sequential data, such as text or time series, which uses self-attention mechanisms to model relationships between data point.  |
|  Data augmentation:  |  A technique used in machine learning to modify the dataset by creating modified versions of existing data points.  |
|  Data normalisation:  |  The process of adjusting and scaling data to a common range or distribution, often to improve the performance of machine learning models.  |



## Get Started

1. Install Python>=3.8, PyTorch 1.9.0.
2. Install requirements.txt.
3. Download data.
4. Train the model. The models can be trained usign the colab notebook *full_test_colab.ipynb*.


## Contact

If you have any question or want to use the code, please contact marcmartinho@gmail.com.

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/MAZiqing/FEDformer/

https://github.com/yuqinie98/PatchTST/tree/main/PatchTST_supervised 

https://github.com/cure-lab/LTSF-Linear 


