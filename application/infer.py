## ECE382N - Computer Performance Evaluation / Benchmark
## GPU Lab -- Application Profiling
## Revision 0.1; January 30th, 2024
## Inference ResNet50 model

#%% Import libraries
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import argparse
from PIL import Image

#%% Argument Parser Section
def parse_args():
    parser=argparse.ArgumentParser(description="ECE382N - Computer Performance Evaluation/Benchmark | GPU Lab | ResNet50 model inference using GPUs.")
    parser.add_argument("--data_dir",       type=str,  action="store",      default="./",         help="Dataset directory path.")
    parser.add_argument("--test_data_dir",  type=str,  action="store",      default="test",       help="Test dataset directory path.")
    parser.add_argument("--checkpoint_dir", type=str,  action="store",      default="checkpoint", help="Checkpoint directory path.")
    parser.add_argument("--num_test",       type=int,  action="store",      default=64,           help="Number of data to test.")
    parser.add_argument("--precision",      type=str,  action="store",      default="fp16",       help="Inference precision.")
    args=parser.parse_args()
    return args

#%% Transform Input Data
def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224,224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out

#%% Prediction Function
def predict_dog_prob_of_single_instance(model, tensor, device, precision):
    if precision == 'fp16':
        in_tensor = tensor.to(torch.float16)
    else:
        in_tensor = tensor
    batch = torch.stack([in_tensor])
    batch = batch.to(device) # Send the input to GPU
    softMax = nn.Softmax(dim = 1)
    preds = softMax(model(batch))
    return preds[0,1].item()

#%% Main Section
def main():
    args = parse_args()
    
    # Detect CUDA-capable Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device=="cpu":
        print("Sadly, I need CUDA-capable device to continue :(")
        exit(1)
        
    # Precision
    if args.precision=='fp32':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        print("Running enforced single-precision (FP32) inference.")
    elif args.precision=='tf32':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        print("Running single-precision (FP32) inference.")
        print("Warning! Since it is not enforced, FP32 maybe demoted to TF32 if GPU supports it.")
    elif args.precision=='fp16':
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        print("Running half-precision (FP16) inference.")
    else:
        print("I do not understand the precision given :(")
        exit(1)
    
    # Download the pre-trained model of ResNet-50
    model_conv  = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Define the checkpoint location to save the trained model
    chk_dir     = f'{args.data_dir}/{args.checkpoint_dir}'
    check_point = f'{chk_dir}/model-checkpoint-{args.precision}.tar'
    
    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # We change the parameter of the final fully connected layer.
    # We have to keep the number of input features to this layer.
    # We change the output features from this layer into 2 features (i.e., we only have two classes).
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    # Load the checkpoint
    checkpoint = torch.load(check_point)
    print("Checkpoint Loaded")
    print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
    model_conv.load_state_dict(checkpoint['model_state_dict'])

    # Precision Adjust
    if args.precision=='fp16':
        model_conv =  model_conv.to(torch.float16)

    # Copy the model into GPU memory
    model_conv = model_conv.to(device)

    # Set the model to eval mode
    model_conv.eval()
        
    # Start the inference
    test_data_files = os.listdir(args.test_data_dir)
        
    image_inferenced   = 0
    for fname in test_data_files :    
        im         = Image.open(f'{args.test_data_dir}/{fname}')
        imstar     = apply_test_transforms(im)    
        outputs    = predict_dog_prob_of_single_instance(model_conv, imstar, device, args.precision)

        if(outputs<0.5) :
            print('Image ' + f'{args.test_data_dir}/{fname}' + ' is predicted as CAT with probability: ' + str(round((1-outputs)*100,2))+ '%')
        else :
            print('Image ' + f'{args.test_data_dir}/{fname}' + ' is predicted as DOG with probability: ' + str(round(outputs*100))+ '%') 
        image_inferenced += 1
        if(image_inferenced>=args.num_test) :
            break

# %% Run Main Section
if __name__=="__main__": 
    main()
