#For road segmentation
import torch
import argparse
import os
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
from torch import nn, optim
from torchvision.transforms import transforms
from torch.optim import lr_scheduler
from ESPNET import ESPNet,ESPNet_Encoder
from dataset import RoadDataset
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize((256,512)), #for onlinereference
    #transforms.Resize((1080,1920)), #for bdd
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5],std=[0.5])
])

# mask只需要转换为tensor

#y_transforms = transforms.ToTensor()
y_transforms = transforms.Compose([
    transforms.Resize((256,512)), #for onlinereference
    #transforms.Resize((1080,1920)),#for ESPnet
    #transforms.ToTensor()
])
def validation(epoch,model, criterion, optimizer, val_loader):
    model.eval()
    step=0
    epoch_loss = 0
    dt_size = len(val_loader.dataset)
    with torch.no_grad():
        for x, y in val_loader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            print("%d/%d,val_loss:%0.5f " % (step, len(val_loader), loss.item()))
        #print("epoch %d val_loss:%0.5f " % (epoch+1, epoch_loss/step))
    return epoch_loss/step


def train_model(model, criterion, optimizer, train_loader, validation_loader, scheduler,num_epochs=1):
    for epoch in range(num_epochs):
        #scheduler.step()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dt_size = len(train_loader.dataset)
        epoch_loss = 0
        step = 0

        #num_correct = 0
        for x, y in train_loader:
            num_correct = 0
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.5f " % (step, len(train_loader), loss.item()))
        
        print("validation... epoch : %d"%(epoch+1))
        val_loss = validation(epoch, model, criterion, optimizer, validation_loader)
        print("epoch %d train_loss:%0.5f val_loss:%0.5f" % (epoch, epoch_loss/step,val_loss))
        
        fp = open("ESPNet_Line_epoch_%d.txt" % num_epochs, "a")
        #fp.write("epoch %d loss:%0.5f \n" % (epoch+1, epoch_loss/step))
        fp.write("epoch %d loss:%0.5f Val_loss:%0.5f\n" % (epoch+1, epoch_loss/step,val_loss))
        fp.close()
        
    torch.save(model.state_dict(), 'ESPNet_Line_weights_epoch_%d.pth' % num_epochs)
    return model

#训练模型
def train(args):
    step_size  = 50
    gamma      = 0.5
    model = ESPNet(12, p=2, q=3).to(device)
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    road_dataset= RoadDataset("./data/training/",transform=x_transforms,target_transform=y_transforms)
    #split training and validation
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(road_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    #dataloaders = DataLoader(road_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(road_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
    validation_loader = DataLoader(road_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
    train_model(model, criterion, optimizer, train_loader, validation_loader,scheduler)

    
def Generate(args):
    model = ESPNet(12, p=2, q=3).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    count=0
    root = './data/testing/'
    print("Generate...")
    import time
    tStart = time.time()
    for filename in os.listdir(root):
        if filename == '.ipynb_checkpoints':
            continue
        imgroot=os.path.join(root+filename)
        name = filename[:-4]
        print("Image Processing: ",name)
        img = Image.open(imgroot)
        img = x_transforms(img)
        img = img.view(1,3,256,512) #forreference
        #img = img.view(1,3,1080,1920) #forreference
        img = img.to(device)
        with torch.no_grad():
            output = model(img).to(device)
            output = torch.softmax(output,dim=1)
            N, _, h, w = output.shape
            pred = output.transpose(0, 2).transpose(3, 1).reshape(-1, 12).argmax(axis=1).reshape(N, h, w) #class 12
            pred = pred.squeeze(0)
            Decode_image(pred,name)
    tEnd = time.time()#計時結束
    #列印結果
    print ("It cost %f sec" % (tEnd - tStart))#會自動做近位
def Model_visualization(args):
    from torchsummary import summary
    #model = Unet(3, 3).to(device)
    model = ESPNet_Encoder(12, p=2, q=3).to(device)
    summary(model, input_size=(3,512,512)) 
def oonx(model):
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    model = ESPNet_Encoder(12, p=2, q=3).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    torch.onnx.export(model, dummy_input, "bbd.onnx", verbose=True)
def Decode_image(img_n,name):
    pallete = [[0,0,0] , 
            [255, 0, 255], [128, 0, 128], [0, 64, 64], [0, 0, 0], [0, 0, 0], 
            [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], 
            [255, 0, 255]]
    img_n = img_n.cpu()
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int) #class 12
    for idx in range(len(pallete)):
        [b, g, r] = pallete[idx]
        img_ans[img_n == idx] = [b, g, r] 
    im_ans = Image.fromarray(np.uint8(img_ans)).convert('RGB') 
    im_ans = cv2.cvtColor(np.array(im_ans),cv2.COLOR_RGB2BGR)         
    #return im_ans           
    cv2.imwrite("./Result/"+name+"_espnet_pred200.png",im_ans)


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
    elif args.action=="model":
        Model_visualization(args)
    elif args.action=="generate":
        Generate(args)
    elif args.action=="oonx":
        oonx(args)
        


