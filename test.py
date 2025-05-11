text = """Class     Images Instances      P     R  mAP50  mAP50-95: 100%
all        583      1581  0.968  0.968  0.986  0.616
open eye   583       607  0.937  0.973  0.98   0.531
open mouth 583        35  0.987  0.943  0.982  0.743
closed eye 583       409  0.962  0.958  0.988  0.56
closed mouth 583       530  0.988  0.998  0.995  0.629
Speed: 0.8ms pre-process, 6.1ms inference per image
Training metrics saved to /home/ubuntu/faster_ssd/datasets/datasets//faster/training_results.csv
Results saved to /home/ubuntu/faster_ssd/datasets/datasets//faster
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool2d'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
[INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
GFLOPS: 37.54, Params: 28289256.00
FPS: 127.22"""

lines = text.split('\n')
for line in lines:
    print(line)