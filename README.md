The Research on Compression and Adversarial Performance of DNN
-----------------------

深度神经网络的网络容量、计算开销、抗扰动能力一直备受关注，当将它部署到嵌入式或移动设备中时这些问题更加突出；攻击者使用一些包含扰动的样本来攻击它，它便会失去原有的可靠性，乃至对错误的抉择持高置信度。本文中，笔者利用交替方向乘子法——其具备各种裁剪方案且与对抗性训练有着很高的兼容性——研究了在对抗性训练环境下的非结构化、过滤器和列剪枝模型压缩方法，分析同时进行剪枝与对抗训练的效果。

Nowadays, deep neural networks can be seen everywhere, nevertheless, its network capacity, computing cost and anti-disturbance ability has been paid much attention by researchers, especially when deep neural networks are deployed into embedded systems or mobile devices, those problems become more obvious. Attackers only use some samples containing perturbation to attack the deep neural network model, the model will lose its originally high reliability and even have a high degree of confidence in the wrong choice. In this work, author used the Alternating Direction Method of Multipliers approach, which has several pruning schemes and high compatibility with adversarial training, to research the irregular pruning, filter pruning and column pruning in the adversarial training setting, analyzing the effect of simultaneously implementing pruning method and adversarial training.

Cite this work:
Shaokai Ye\*, Kaidi Xu\*, Sijia Liu, Hao Cheng, Jan-Henrik Lambrechts, Huan Zhang, Aojun Zhou, Kaisheng Ma, Yanzhi Wang, Xue Lin. ["Adversarial Robustness vs Model Compression, or Both?"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf), ICCV 2019. (\* Equal Contribution)

and

Shaokai Ye, Xiaoyu Feng, Tianyun Zhang, Xiaolong Ma, Sheng Lin, Zhengang Li, Kaidi Xu, Wujie Wen, Sijia Liu, Jian Tang, Makan Fardad, Xue Lin, Yongpan Liu, Yanzhi Wang. ["Progressive DNN Compression: A Key to Achieve Ultra-High Weight Pruning and Quantization Rates using ADMM"](https://arxiv.org/pdf/1903.09769.pdf), arXiv:1903.09769


Train a model in natural setting/adversarial setting
-----------------------


'main.py' or 'adv_main.py' for main program, natural setting and adversarial setting respectively

'eval.py' for quick checking of the sparsity and do some other stuff (like attack test, etc.)

'config.yaml' template of the configuration file. One for each dataset.

'run.sh.example' template script for running the code.



Compression in adversarial setting are only supported for MNIST and CIFAR10. 


Prerequisites
-----------------------

code is compatible with pytorch 1.0.0
