# mnist.py 다운로드
wget https://raw.githubusercontent.com/kubeflow/katib/master/examples/v1beta1/trial-images/pytorch-mnist/mnist.py

# ConfigMap 생성
kubectl create configmap pytorch-mnist-code --from-file=mnist.py -n jhchoi
