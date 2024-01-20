<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

# MedIA 2024 | 多器官的非连续性形变配准 —— 深度学习网络设计中的运动可分与解耦 #

在医学领域，图像配准是一项关键技术，它涉及将不同时间或不同模态扫描的图像进行精准对齐。这在腹部器官的成像分析中尤为重要，因为腹部器官常常会因为呼吸、消化等生理活动而移动或形变。牛津大学医学图像研究团队探讨了在腹部医学影像配准中遇到的一个尚未解决的难题：多器官间的非连续滑动或分离运动。这些运动的复杂性为准确识别和跟踪各器官带来了困难。

本文的研究重点是解决多器官非连续形变配准问题。文中提出了一个创新的量化指标 —— 运动可分离度（Motion Separability）。这个指标旨在衡量深度学习网络预测器官非连续形变的能力。基于这一指标，文中设计了一种新型的网络结构骨架 —— 运动分离骨架（Motion-Separable structure）。此外，我们还引入了运动解耦模块（Motion disentanglement），帮助网络区分并处理不同器官间的复杂运动模式。

为了验证这一量化指标的有效性以及我们方法的准确性和高效性，文中进行了一系列的非监督配准实验。这些实验涵盖了腹部的九个主要器官及肺部图像。实验结果显示，文中的方法不仅能够有效识别和处理各器官间的复杂运动，还提高了配准的准确性和效率。

* 其中的主要贡献包括：
  - **非连续配准网络**：这是第一个基于深度学习网络的针对非连续性形变配准的定量研究。
  - **理论分析**： 本文量化并定义了神经网络中的最大可捕获运动范围和运动可分离上界，为运动可分离的上限提供了理论分析。这有助于我们理解网络能识别的运动范围，指导网络结构和参数设置的优化。
  - **运动可分的骨架结构**：基于本文的理论分析，文中设计了一个新型的多尺度骨架结构。这个结构通过在高分辨率特征图上使用优化的扩张卷积，使网络能够有效预测更大可分离上界的运动模式，同时保持可捕获的运动范围与较低的计算复杂度。
  - **运动解耦和细化模块**：此外，我们提出的残差对齐模块（Residual Aligner module），利用置信度和基于语义及上下文信息的机制，来区分不同器官或区域中预测的位移。这意味着我们的方法可以更准确地处理每个区域的特定运动。
  - **准确且高效的配准结果**：上述所提出的组件构成了新颖的残差对齐网络（RAN），在公开的肺部和腹部CT数据上执行高效、粗到细、可分离运动的无监督配准，取得了更高的精确度和更低的计算成本。

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/a2649412-6a58-4d7a-b919-24a7ca89af05)

* 论文发表于 Medical Image Analysis：https://doi.org/10.1016/j.media.2023.103038
* 代码即将开源于: https://github.com/jianqingzheng/res_aligner_net



## 研究背景 ##

在医学成像领域，变形图像配准——即估算不同图像间的空间变换——是一项重要任务。
现有的深度学习配准网络可以大致分类为：直接拟合网络（direct regression，如voxelmorph[1]），基于注意力机制网络（attention-based network，如transmorph[2]），级联网络（Cascaded network，如Recursive Cascaded network[3] 和 Composite-net[4]），和特征图金字塔网络（Feature Pyramid，如Dual-PRNet[5]）

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/ffb20958-abf7-4391-bc39-b2fe65ef45de)


深度学习技术已被证明可以高效地进行三维图像配准。然而，目前的配准策略往往只关注形变的平滑性，从而忽略了复杂的运动模式（如分离运动或滑动运动），尤其是器官的交汇处。因此，在处理附近多个物体、器官的不连续运动时，预测的形变场中会出现相近器官的粘连问题，从而限制配准网络预测的性能上限，在临床应用中会造成不理想的预测结果，如病变或其他异常的误识别和误定位。

因此，本文提出了一种新颖的深度学习网络配准方法（Residual Aligner Network），专门解决这一问题：采用新型的运动可分网络骨架（Motion-Separable structures, MS structure）来捕捉分离运动，并通过残差对齐模块（Residual Aligner module）对多个相邻物体/器官的预测运动进行解耦和精细化​​。

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/190d79e5-da8b-412b-932b-66b3008c61c9)



## 运动可分网络骨架设计 ##
我们首先介绍了有粗到细（Coarse-to-fine）的配准框架，然后分析和量化了粗细配准网络在捕捉大变形（可捕捉运动范围，Accessible motion range）和保持不连续性（预测运动的可分性，Motion Separability）方面的能力。为增强网络在保持大不连续性的同时捕捉大变形的能力，我们提出了一种新的运动可分网络骨架设计方法，运动可分全卷积网络（MSFCN）。该网络骨架通过升采样来得到相对高分别率的稠密形变位移场（Dense Displacement field, DDF），并用空洞卷积（Atrous/dilated convolution）来提取特征图以保证足够大的感受野（Receptive field）,从而保证足够大的可捕捉运动范围。其中可分离度模式（MS pattern）层数```q```用来调整每层的池化大小```p_k```与空洞率```r_k```，从而保证在相同的可捕捉运动范围情况下，```q```越大的网络获得更大的预测运动的可分性。

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/203dfeb7-29ba-485f-bf52-3e2a1826e151)

## 可分性上界理论分析 ##

在文中将该神经网络所预测运动的可分性定义为：

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/79b6f3a3-b04c-4d37-a549-d28e885292ad)

通过理论分析得到预测运动的可分性上界与池化大小```p_k```与空洞率```r_k```的相关性（$ a_k\approx p_k(1+2||r_k||_1)/2 $）

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/7a604569-4a84-42a5-9652-40a754a92a51)

![image](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/bfd419ae-6bd9-4728-b9e1-62ae0a38f536)


## 实验结果 ##

![deformable_reg-min](https://github.com/jianqingzheng/res_aligner_net/assets/39138328/ffecee77-bdda-4ac8-8150-68a1426d8bff)


## 参考文献
