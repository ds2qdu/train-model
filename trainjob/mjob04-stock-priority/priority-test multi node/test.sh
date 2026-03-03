kubectl apply -f 01-resources.yaml
kubectl apply -f 02-pvc.yaml
kubectl apply -f 03-secret.yaml
kubectl apply -f 04-runtime.yaml
kubectl apply -f 05-trainjob-low.yaml
sleep 30
kubectl apply -f 06-trainjob-high.yaml