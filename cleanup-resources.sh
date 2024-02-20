#!/bin/bash

# Stop all Docker containers
docker stop $(docker ps -aq)

# Remove all Docker containers
docker rm $(docker ps -aq)

# Remove all Docker images
docker rmi $(docker images -q)

# Remove all Docker volumes
docker volume rm $(docker volume ls -q)

# Switch to the Kubernetes context you want to clean up
# kubectl config use-context your-context-name

# Delete all Kubernetes pods in all namespaces
kubectl delete pods --all --all-namespaces

# Delete all Kubernetes services in all namespaces (excluding the default kubernetes service in the default namespace)
kubectl delete services --all --all-namespaces --ignore-not-found=true
kubectl delete services --all --namespace=default --ignore-not-found=true --field-selector=metadata.name!=kubernetes

# Delete all Kubernetes persistent volume claims and persistent volumes in all namespaces
kubectl delete pvc --all --all-namespaces
kubectl delete pv --all --all-namespaces

echo "All Docker and Kubernetes resources have been deleted."
