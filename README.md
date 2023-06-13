# TRANSFORMER-BASED DEEP HASHING METHOD FOR MULTI-SCALE FEATURE（TDH）(ICASSP 2023)

The code used in this paper references existing code from [huggingface](https://github.com/huggingface/pytorch-image-models) and [swuxyj](https://github.com/swuxyj/DeepHash-pytorch).

Paper Link
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10094794

Method
-----
![TDH](https://user-images.githubusercontent.com/49743419/220228569-dcd3c9d5-33e9-49de-bec0-fcfb17b8e5d2.png)

We also provide performance and efficiency comparisons with different backbone networks,Table 1 presents the performance results of the
different methods on the three benchmark datasets where TDH-R101 indicates the use of the
ResNet101 backbone network and TDH-V-L-16 indicates the use of the ViT-L_16 backbone
network. Table 2 demonstrates the number of parameters and the computational effort of the
model under different backbone networks.Table 3 illustrates the architecture detail of TDH.

![new](https://user-images.githubusercontent.com/49743419/220231454-b6e2bdf1-1b52-4293-b28f-d6329926c6cc.png)
![捕获](https://user-images.githubusercontent.com/49743419/221077773-7e4b9e5f-233f-4dcf-a600-133bdb97bee4.png)

Training
-----
All parameters are defined in the TDH_train.py file, and the test methods are integrated into the TDH_train.py file.Therefore, it is only necessary to run

python TDH_train.py

Datasets
-----
To download the dataset, please visit the [swuxyj](https://github.com/swuxyj/DeepHash-pytorch).
