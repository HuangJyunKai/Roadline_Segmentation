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
from DataSet import MyDataset
import Transforms as myTransforms
from IOUEval import iouEval
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean=[90.33802,92.7675,93.35951]
std=[57.88047,57.541573,57.865982]
classweights = np.array([ 1.4102876,10.443367,10.478665 , 10.480342 ,  8.17369  , 10.392894,
  9.908393 , 10.019764 , 10.172403 , 10.418645  , 9.616669  , 9.481122 ])
classweights = torch.from_numpy(classweights).float().to(device)
trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(512, 256),
        #myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(1),
        #
    ])

trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(1536, 768), # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(1),
        #
    ])

trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(1280, 720), # 1536, 768
        myTransforms.RandomCropResize(100),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(1),
        #
    ])

trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(1024, 512),
        myTransforms.RandomCropResize(32),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64),
        myTransforms.ToTensor(1),
        #
    ])

trainDataset_scale4 = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(768, 384),
        myTransforms.RandomCropResize(20),
        myTransforms.RandomFlip(),
        #myTransforms.RandomCrop(64).
        myTransforms.ToTensor(1),
        #
    ])


valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean, std),
        myTransforms.Scale(512, 256),
        myTransforms.ToTensor(1),
        #
    ])
nclass=12 # 0 background
IGNORE_LABEL = 0
def validation(epoch,model, criterion, optimizer, val_loader):
    iouEvalVal = iouEval(nclass)
    model.eval()
    step=0
    epoch_loss = 0
    epoch_mIOU = 0
    epoch_acc = 0. 
    dt_size = len(val_loader.dataset)
    with torch.no_grad():
        for x, y in val_loader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            outputs = torch.softmax(outputs,dim=1)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            print("%d/%d,val_loss:%0.5f " % (step, len(val_loader), loss.item()))
            #mIOU
            
            iouEvalVal.addBatch(outputs.max(1)[1].data, labels.data)
        overall_acc, per_class_acc, per_class_iou, mIOU = iouEvalVal.getMetric()
    print("epoch %d val_loss:%0.5f " % (epoch+1, epoch_loss/step))
    print("overall_acc :",overall_acc)
    print("per_class_acc :",per_class_acc)
    print("per_class_iou :",per_class_iou)
    print("mIOU :",mIOU)
    return epoch_loss/step , overall_acc, per_class_acc, per_class_iou, mIOU


def train_model(model, criterion, optimizer, train_loader, scheduler, epoch, num_epochs):
    iouEvalTrain = iouEval(nclass)
    #scheduler.step()
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
        outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
            
        
        iouEvalTrain.addBatch(outputs.max(1)[1].data, labels.data)
            
        print("%d/%d,train_loss:%0.5f " % (step, len(train_loader), loss.item()))
    overall_acc, per_class_acc, per_class_iou, mIOU = iouEvalTrain.getMetric()
    print("overall_acc :",overall_acc)
    print("per_class_acc :",per_class_acc)
    print("per_class_iou :",per_class_iou)
    print("mIOU :",mIOU)
    dirName = "./models/ESPNet_Line_mytransform_256_512_epoch150/"
    if not os.path.exists(dirName):
       os.mkdir(dirName)
       print("Directory " , dirName ,  " Created ")        
    torch.save(model.state_dict(), dirName+'ESPNet_Line_mytransfrom_256_512_weights_epoch_%d.pth' % (epoch+1))
    return epoch_loss/step, overall_acc, per_class_acc, per_class_iou, mIOU

#训练模型
def train(args):
    num_epochs = 150
    step_size  = 50
    gamma      = 0.5
    model = ESPNet(12, p=2, q=3).to(device)
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
    #criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL,reduction='mean')
    criterion = nn.CrossEntropyLoss(weight=classweights)
    road_dataset= MyDataset("./data/training/",transform=trainDataset_main)
    road_dataset_scale1 = MyDataset("./data/training/",transform=trainDataset_scale1)
    road_dataset_scale2 = MyDataset("./data/training/",transform=trainDataset_scale2)
    road_dataset_scale3 = MyDataset("./data/training/",transform=trainDataset_scale3)
    road_dataset_scale4 = MyDataset("./data/training/",transform=trainDataset_scale4)
    road_dataset_val    = MyDataset("./data/training/",transform=valDataset)
    #split training and validation
    validation_split = .2
    shuffle_dataset = True
    dataset_size = len(road_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        #np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    #multi-scale dataloader
    train_loader = DataLoader(road_dataset, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale1 = DataLoader(road_dataset_scale1, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale2 = DataLoader(road_dataset_scale2, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale3 = DataLoader(road_dataset_scale3, batch_size=batch_size, sampler=train_sampler)
    train_loaderscale4 = DataLoader(road_dataset_scale4, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(road_dataset_val, batch_size=batch_size,sampler=valid_sampler)
    
    for epoch in range(num_epochs):
        print("epoch: %d/%d" %(epoch+1,num_epochs)) 
        print("scale : 1536x768")
        train_model(model, criterion, optimizer, train_loaderscale1, scheduler, epoch, num_epochs)
        print("scale : 1280x720")
        train_model(model, criterion, optimizer, train_loaderscale2, scheduler, epoch, num_epochs)
        print("scale : 1024x512")
        train_model(model, criterion, optimizer, train_loaderscale3, scheduler, epoch, num_epochs)
        print("scale : 768x384")
        train_model(model, criterion, optimizer, train_loaderscale4, scheduler, epoch, num_epochs)
        print("scale : 512x256")
        epoch_loss, overall_acc, per_class_acc, per_class_iou, mIOU = train_model(model, criterion, optimizer, train_loader,scheduler, epoch, num_epochs)
        print("scale : 512x256")
        valepoch_loss,valoverall_acc, valper_class_acc, valper_class_iou, valmIOU = validation(epoch, model, criterion, optimizer, validation_loader)
        
        fp = open("ESPNet_Line_mytransform_256_512_epoch_%d.txt" % num_epochs, "a")
        fp.write("epoch %d train_loss:%0.3f \n" % (epoch+1, epoch_loss))
        fp.write("train overall_acc:%0.3f \n"%(overall_acc))
        fp.write("train per_class_acc: ")
        fp.write(str(per_class_acc))
        fp.write("\n")
        fp.write("train per_class_iou: ")
        fp.write(str(per_class_iou))
        fp.write("\n")
        fp.write("train mIOU:%0.3f\n"% (mIOU))
        fp.write("epoch %d val_loss:%0.3f \n" % (epoch+1, valepoch_loss))
        fp.write("val overall_acc:%0.3f \n" % (valoverall_acc))
        fp.write("val per_class_acc: ")
        fp.write(str(valper_class_acc))
        fp.write("\n")
        fp.write("vak per_class_iou: ")
        fp.write(str(valper_class_iou))
        fp.write("\n")
        fp.write("val mIOU:%0.3f\n"% (valmIOU))
        fp.write("\n")
        fp.close()
def Val(args):
    model = ESPNet(12, p=2, q=3).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    road_dataset= RoadDataset("./data/training/",transform=x_transforms,target_transform=y_transforms)
    validation_split = .2
    shuffle_dataset = True
    dataset_size = len(road_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    valid_sampler = SubsetRandomSampler(val_indices)
    validation_loader = DataLoader(road_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
    overall_acc, per_class_acc, per_class_iu, mIOU = validation(1, model, criterion, optimizer, validation_loader)
    print("overall_acc:%0.3f per_class_acc:%0.3f per_class_iu:%0.3f, mIOU:%0.3f"%(overall_acc, per_class_acc, per_class_iu, mIOU))
def Generate(args):
    model = ESPNet(12, p=2, q=3).to(device)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    model.eval()
    count=0
    root = './data/testing/'
    print("Generate...")
    import time
    tStart = time.time()
    #mean = [72.3923111, 82.90893555, 73.15840149]
    #std = [45.3192215, 46.15289307, 44.91483307]
    #mean = [0.485, 0.456, 0.406]
    #std=[0.229, 0.224, 0.225]
    
    for filename in os.listdir(root):
        if filename == '.ipynb_checkpoints':
            continue
        imgroot=os.path.join(root+filename)
        name = filename[:-4]
        print("Image Processing: ",name)
        img = cv2.imread(imgroot)
        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]
        img = cv2.resize(img, (512, 256))
        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        img_tensor = img_tensor.to(device)
        
        output = model(img_tensor)
        #output = torch.softmax(output,dim=1)
        pred = output[0].max(0)[1].byte().cpu().data.numpy()
        print(np.unique(pred))
        Decode_image(pred,name)
    tEnd = time.time()#計時結束
    #列印結果
    print ("It cost %f sec" % (tEnd - tStart))#會自動做近位

def Decode_image(img_n,name):
    pallete = [[0,0,0] , 
            [255, 0, 255], [128, 0, 128], [0, 64, 64], [0, 0, 0], [0, 0, 0], 
            [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], 
            [255, 0, 255]]
    img_ans=np.zeros((img_n.shape[0],img_n.shape[1],3), dtype=np.int) #class 12
    for idx in range(len(pallete)):
        [b, g, r] = pallete[idx]
        img_ans[img_n == idx] = [b, g, r] 
    im_ans = Image.fromarray(np.uint8(img_ans)).convert('RGB') 
    im_ans = cv2.cvtColor(np.array(im_ans),cv2.COLOR_RGB2BGR)         
    #return im_ans           
    cv2.imwrite("./Result/"+name+"_espnet_pred_27_mytransform.png",im_ans)
if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=16)
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
    elif args.action=="check":
        check_label(args)
    elif args.action=="iou":
        Val(args)
        


