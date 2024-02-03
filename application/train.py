## ECE382N - Computer Performance Evaluation / Benchmark
## GPU Lab -- Application Profiling
## Revision 0.1; January 30th, 2024
## Fine-tuning ResNet50 model

#%% Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
import datetime
import time
import copy
import math
import argparse
from tqdm import tqdm

# %% Training flow
def train_model(model, criterion, optimizer, scheduler, num_epochs, timestamp, device, dataloaders, dataset_sizes, amp, profiling):
    since = time.time()
    
    # Initialization
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_acc = 0.
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print('Running epoch {} from {} epochs'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        if not profiling:
            phase_list = ['train', 'val']
        else:
            phase_list = ['train']
            
        for phase in phase_list:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autocast(device_type="cuda",enabled=amp, dtype=torch.float16, cache_enabled=True):
                       outputs = model(inputs)
                       _, preds = torch.max(outputs, 1)
                       loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if amp==True:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print(f'New best model found!')
                print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc

#%% Argument Parser Section
def parse_args():
    parser=argparse.ArgumentParser(description="ECE382N - Computer Performance Evaluation/Benchmark | GPU Lab | ResNet50 model finetuning using GPUs.")
    parser.add_argument("--num_epoch",      type=int,  action="store",      default=15,           help="Number of epoch for training to run.")
    parser.add_argument("--batch_size",     type=int,  action="store",      default=1000,         help="Batch size for training.")
    parser.add_argument("--num_worker",     type=int,  action="store",      default=64,           help="Number of data loader worker.")
    parser.add_argument("--precision",      type=str,  action="store",      default="fp16",       help="Training precision.")
    parser.add_argument("--profile",                   action="store_true", default=False,        help="Disable validation and checkpoint storage of the model, useful for profiling replay.")
    args=parser.parse_args()
    return args

#%% Main Section
def main():
    args = parse_args()
    
    # Root Folder
    data_dir       = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = 'checkpoint'
    
    # Detect CUDA-capable Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device=="cpu":
        print("Sadly, I need CUDA-capable device to continue :(")
        exit(1)
        
    # Precision
    if args.precision=='fp32':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        print("Running enforced single-precision (FP32) training.")
        amp = False
    elif args.precision=='tf32':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        print("Running single-precision (FP32) training.")
        print("Warning! Since it is not enforced, FP32 maybe demoted to TF32 if GPU supports it.")
        amp = False
    elif args.precision=='fp16':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        print("Running half-precision (FP16) training.")
        amp = True
    else:
        print("I do not understand the precision given :(")
        exit(1)

    if args.profile:
        print("Disabling model checkpoint and validation since profiling mode is active.")
        
    # Define dataset transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Define the data transformation
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    
    # Define the data loader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_worker)
                  for x in ['train', 'val']}
    
    # Print the statistics
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)
    print(f'Train image size: {dataset_sizes["train"]}')
    print(f'Validation image size: {dataset_sizes["val"]}')
    
    # Download the pre-trained model of ResNet-50
    model_conv  = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Define the checkpoint location to save the trained model
    chk_dir     = f'{data_dir}/{checkpoint_dir}'
    check_point = f'{chk_dir}/model-checkpoint-{args.precision}.tar'
    
    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # We change the parameter of the final fully connected layer.
    # We have to keep the number of input features to this layer.
    # We change the output features from this layer into 2 features (i.e., we only have two classes).
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    # Copy the model into GPU memory
    model_conv = model_conv.to(device)
    
    # Choose the Criterion as Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimize only the parameters of the final fully connected layer since we have changed them.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    # This is our learning rate scheduler. Decay learning rate by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    # Training Loop
    today = datetime.datetime.today() 
    timestamp = today.strftime('%Y%m%d-%H%M%S')
    
    # Start the training
    model_conv, best_val_loss, best_val_acc = train_model(model_conv,
                                                          criterion,
                                                          optimizer_conv,
                                                          exp_lr_scheduler,
                                                          args.num_epoch,
                                                          timestamp,
                                                          device,
                                                          dataloaders,
                                                          dataset_sizes,
                                                          amp,
                                                          args.profile)
    # Save the trained model for future use.
    if not args.profile:
        torch.save({'model_state_dict': model_conv.state_dict(),
                    'optimizer_state_dict': optimizer_conv.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_acc,
                    'scheduler_state_dict' : exp_lr_scheduler.state_dict(),
                    }, check_point)

# %% Run Main Section
if __name__=="__main__": 
    main()
