python==3.7

[tensorflow 의존성 확인](https://www.tensorflow.org/install/source_windows)  
cudnn==7.6
cuda==10.1



# 의존성 설치
## tensorflow, pytorch
```text
# pip
pip install tensorflow==2.3.4 tensorflow-gpu==2.3.4 transformers==3.3.1 pandas seqeval fastprogress attrdict

# conda
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

## onnx
```text
pip install tf2onnx==1.12.0
pip install onnxruntime==1.12.1
```