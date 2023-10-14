
# DYNCONV - CLASSIFICATION (CIFAR10,CIFAR100,SVHN,IMAGENET)

## Requirements->
```javascript
pip install -r requirements.txt
```
Note: Specify the path of your imagenet dataset using the --dataset-root argument, or change the path in the main_imagenet.py file.
Also change the model according to your requirements. 


## Model Run Commands
Different model files for the particular models have been shared in this repository.

#### Resnet32 on Cifar10
To train model on cifar10 dataset, run the below command ->
```javascript
python main_cifar10.py --model resnet32 --save_dir exp/your_new_run -r exp/your_new_run --budget 0.1
```

To evaluate on cifar10 dataset, run the below command ->
```javascript
python main_cifar10.py --model resnet32 -r exp/cifar/resnet32/sparse01/checkpoint_best.pth -e
```
This should output:
>\* Epoch 300 - Prec@1 90.08
>\* FLOPS (multiply-accumulates, MACs) per image:  13.53 MMac

#### Resnet32 on Cifar100
To train model on cifar100 dataset, run the below command ->
```javascript
python main_cifar100.py --model resnet32 --save_dir exp/your_new_run -r exp/your_new_run --budget 0.9
```

To evaluate on cifar100 dataset, run the below command ->
```javascript
python main_cifar100.py --model resnet32 -r exp/cifar/resnet32/sparse09/checkpoint_best.pth -e
```
This should output:
>\* Epoch 300 - Prec@1 70.21
>\* FLOPS (multiply-accumulates, MACs) per image:  67.41 MMac

#### Resnet32 on SVHN
To train model on SVHN dataset, run the below command ->
```javascript
python main_SVHN.py --model resnet32 --save_dir exp/your_new_run -r exp/your_new_run --budget 0.5
```

To evaluate on SVHN dataset, run the below command ->
```javascript
python main_SVHN.py --model resnet32 -r exp/cifar/resnet32/sparse05/checkpoint_best.pth -e
```
This should output:
>\* Epoch 300 - Prec@1 98.95
>\* FLOPS (multiply-accumulates, MACs) per image:  36.18 MMac

#### Resnet32 on Imagenet
To train model on Imagenet dataset, run the below command ->
```javascript
python main_imagenet.py --model resnet32 --save_dir exp/your_new_run -r exp/your_new_run --budget 0.3
```

To evaluate on Imagenet dataset, run the below command ->
```javascript
python main_imagenet.py --model resnet32 -r exp/cifar/resnet32/sparse04/checkpoint_best.pth -e
```
This should output:
>\* Epoch 300 - Prec@1 74.89
>\* FLOPS (multiply-accumulates, MACs) per image:  2997.18 MMac



##### For any doubt or discussion, you can connect with me on bhipanshudhupar@gmail.com or www.linkedin.com/in/bhipanshu-dhupar






