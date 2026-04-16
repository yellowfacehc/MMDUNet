import os
import torch
import time
## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

kfold = 5
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 300
img_size = 224
img_size2 = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000


task_name = 'NIH_Pancreas'


if task_name is 'NIH_Pancreas':
    learning_rate = 1e-3
    batch_size = 16
    early_stopping_patience = 40
    print_frequency = 5

model_name = 'MMDUNet'


if task_name is "NIH_Pancreas":
    if model_name is "MMDUNet":
        test_session = "Test_session_07.24_19h48"

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'


session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'_kfold/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'