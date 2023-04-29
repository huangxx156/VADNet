# Boosting Variational Inference with Margin Learning for Few-Shot Scene-Adaptive Anomaly Detection

This is the code for ["Boosting Variational Inference with Margin Learning for Few-Shot Scene-Adaptive Anomaly Detection".](https://ieeexplore.ieee.org/document/9976040)

## Prerequisites
- Python 3.7
- Pytoch 1.7.1
- Numpy
- matplotlib

## Datasets
Download the datasets into ``dataset`` folder, like ``./dataset/test/``

the data directory should be structured like this:

  ```
   |-- test
       |-- norm
           |-- ped2
               |-- 01
               |-- 02
           |-- ...
       |-- notes
           |-- ped2
           |-- ...
  ```
## Evaluation
* Download our pre-trained model: [Link1](https://drive.google.com/file/d/13s5cAu4VbIP_TOa8bmT4SHe-hub96Yug/view?usp=share_link) or [Link2](https://pan.baidu.com/s/1DByWyR--KIY6MIgzwC5HBQ) (3bf1)
* The model is trained on the Shanghai Tech and test on the UCSD Ped2.

* Test the model with our pre-trained model
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --exp_name test --pretrained --pretrained_path best_proto_ped2.tar  --dataset_path ./dataset --set ped2
```

## Bibtex
```
@article{huang2022boosting,
  title={Boosting Variational Inference with Margin Learning for Few-Shot Scene-Adaptive Anomaly Detection},
  author={Huang, Xin and Hu, Yutao and Luo, Xiaoyan and Han, Jungong and Zhang, Baochang and Cao, Xianbin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```


