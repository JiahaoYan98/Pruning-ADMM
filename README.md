The Research on Compression and Adversarial Performance of DNN
-----------------------

In this work, we use a framework of concurrent adversarial training and different weight pruning that enables model compression while still preserving the adversarial robustness and essentially tackles the dilemma of adversarial training.  

Cite this work:
Shaokai Ye\*, Kaidi Xu\*, Sijia Liu, Hao Cheng, Jan-Henrik Lambrechts, Huan Zhang, Aojun Zhou, Kaisheng Ma, Yanzhi Wang, Xue Lin. ["Adversarial Robustness vs Model Compression, or Both?"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf), ICCV 2019. (\* Equal Contribution)

and

Shaokai Ye, Xiaoyu Feng, Tianyun Zhang, Xiaolong Ma, Sheng Lin, Zhengang Li, Kaidi Xu, Wujie Wen, Sijia Liu, Jian Tang, Makan Fardad, Xue Lin, Yongpan Liu, Yanzhi Wang. ["Progressive DNN Compression: A Key to Achieve Ultra-High Weight Pruning and Quantization Rates using ADMM"](https://arxiv.org/pdf/1903.09769.pdf), arXiv:1903.09769


Prerequisites
-----------------------

code is compatible with pytorch 1.0.0



Train a model in natural setting/adversarial setting
-----------------------


'main.py' or 'adv_main.py' for main program, natural setting and adversarial setting respectively

'eval.py' for quick checking of the sparsity and do some other stuff (like attack test, etc.)

'config.yaml' template of the configuration file. One for each dataset.

'run.sh.example' template script for running the code.





Compression in adversarial setting are only supported for MNIST and CIFAR10. 



