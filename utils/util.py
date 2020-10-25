from sklearn.metrics import f1_score
import torch

def get_device(gpu_id):

    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")
                          
    n_gpu = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        print("device is cuda, # cuda amount is: ", n_gpu)
        
    else:
        print("device is cpu, not recommend")
        
    return device, n_gpu


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    return acc, f1