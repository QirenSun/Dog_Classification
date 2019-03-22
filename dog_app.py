


import os
from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt 

train_dir=os.path.join('/data/dog_images/train')
test_dir=os.path.join('/data/dog_images/test')
valid_dir=os.path.join('/data/dog_images/valid')
batch_size=20
data_transforms = {
    'train' : transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'valid' : transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test' : transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}


image_datasets = {
    'train' : datasets.ImageFolder(root=train_dir,transform=data_transforms['train']),
    'valid' : datasets.ImageFolder(root=valid_dir,transform=data_transforms['valid']),
    'test' : datasets.ImageFolder(root=test_dir,transform=data_transforms['test']),
}

data_loaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'],batch_size = batch_size,shuffle=True),
    'valid' : torch.utils.data.DataLoader(image_datasets['valid'],batch_size = batch_size),
    'test' : torch.utils.data.DataLoader(image_datasets['test'],batch_size = batch_size)    
}

print("Size of training set is: "+str(len(image_datasets['train'])))
print("Size of validation set is: "+str(len(image_datasets['valid'])))
print("Size of testing set is: "+str(len(image_datasets['test'])))





import torch.optim as optim


criterion_scratch = nn.CrossEntropyLoss


#optimizer_scratch =optim.SGD(model_scratch.parameters(), lr=0.001)
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=7, gamma=0.1)





import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))



import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()



# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


#  Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  


from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## on the images in human_files_short and dog_files_short.
human_detections = np.sum([face_detector(img) for img in tqdm(human_files_short)])
dog_detections = np.sum([face_detector(img) for img in tqdm(dog_files_short)])

print('face detection in human image set = {}%'.format(human_detections))
print('face detection in dog image set = {}%'.format(dog_detections))




import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()


# Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

# Making Predictions with a Pre-trained Model
# 
# In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.
# 
# Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).




from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    img_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    img=Image.open(str(img_path))
    image=img_transform(img)
    image=np.expand_dims(image,axis=0)
    image=torch.from_numpy(image).to('cuda')
    VGG16.eval()
    output=VGG16(image)
    probability=torch.exp(output)
    top_probability,top_class=probability.topk(1,dim=1)
    
    
    
    return top_class.item() # predicted class index


# ###  Write a Dog Detector
# 
# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
# 
# Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

# In[97]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    detector=VGG16_predict(img_path)
    
    return detector>=151 & detector<=268  # true/false


# ###  Assess the Dog Detector
# 

### on the images in human_files_short and dog_files_short.
### on the images in human_files_short and dog_files_short.
import numpy as np
from glob import glob


human_images = np.array(glob("/data/lfw/*/*"))
dog_images = np.array(glob("/data/dog_images/*/*/*"))

# load filenames for human and dog images
#human_images = np.array(glob("lfw/*/*"))
#dog_images = np.array(glob("dogImages/*/*/*"))


n,m=0,0
for i in range(len(human_images[:100])):
    if dog_detector(human_images[i])==True:
        n+=1
for i in range(len(dog_images[:100])):
    if dog_detector(dog_images[i])==True:
        m+=1
Perh=n/len(human_images)
Perd=m/len(dog_images)
print('percentage of the images in human_files have a detected human face:',Perh)
print('percentage of the images in dog_files have a detected human face:',Perd)


# We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

# In[ ]:





import os
from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np 

## Specify appropriate transforms, and batch_sizes
data_dir=os.path.join('/data/dog_images')
transform = transforms.Compose([
        transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  

num_workers = 0
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'valid', 'test']}
    
loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 20,
                                              shuffle = True, num_workers = num_workers) for x in ['train', 'valid', 'test']}



use_cuda = torch.cuda.is_available()





def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)    
    plt.imshow(inp)
    
# Get a batch of training data
images, classes = next(iter(loaders['train']))
class_names = image_datasets['train'].classes
n_classes = len(class_names)      
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(class_names[classes[idx]].split(".")[1])





images.shape




import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1=nn.Conv2d(3,16,3,padding=1)
        self.batch_norm1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        self.batch_norm2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        self.batch_norm3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,128,3,padding=1)
        self.batch_norm4=nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(128,256,3,padding=1)
        self.batch_norm5=nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2=nn.Linear(512,133)
        
        
    def forward(self, x):
        ## Define forward behavior
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=self.batch_norm1(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=self.batch_norm2(x)
        x=F.relu(self.conv3(x))
        x=self.pool(x)
        x=self.batch_norm3(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x=self.batch_norm4(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x=self.batch_norm5(x)
        x=x.view(-1,256*7*7)
        x=self.dropout(x)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x


# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()


# In[40]:


print(model_scratch)



import torch.optim as optim

criterion_scratch = nn.CrossEntropyLoss()

# specify optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9)


# ### (IMPLEMENTATION) Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.

# In[42]:


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data - train_loss)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
# train the model
n_epochs = 5
model_scratch = train(n_epochs, loaders, model_scratch, optimizer_scratch,
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
#model_scratch.load_state_dict(torch.load('model_scratch.pt'))


# ###  Test the Model
# 



def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders, model_scratch, criterion_scratch, use_cuda)




import os
from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt 


## Specify appropriate transforms, and batch_sizes
data_dir=os.path.join('/data/dog_images')
transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  

num_workers = 0
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'valid', 'test']}
    
loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 20,
                                              shuffle = True, num_workers = num_workers) for x in ['train', 'valid', 'test']}



use_cuda = torch.cuda.is_available()


# ###  Model Architecture
# 
# Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.




import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
model_transfer=models.densenet121(pretrained=True)
for p in model_transfer.parameters():
    p.requires_grad=False
num = model_transfer.classifier.in_features
model_transfer.classifier = nn.Linear(num, n_classes)
if use_cuda:
    model_transfer = model_transfer.cuda()




criterion_transfer =nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.classifier.parameters(), lr=0.001, momentum=0.9)


# ###  Train and Validate the Model
# 
# Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.

# In[48]:


# train the model
n_epochs=5
model_transfer =  train(n_epochs, loaders, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))


# ###  Test the Model
# 
# Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.

# In[50]:


test(loaders, model_transfer, criterion_transfer, use_cuda)


# ###Predict Dog Breed with the Model
# 
# Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  



### and returns the dog breed that is predicted by the model.
from torch.autograd import Variable
# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in image_datasets['train'].classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img=Image.open(img_path)
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor=transform(img).float()
    img_tensor.unsqueeze_(0)
    img_tensor = Variable(img_tensor)
    if use_cuda:
        img_tensor = Variable(img_tensor.cuda())
    model_transfer.eval()
    output = model_transfer(img_tensor)
    output = output.cpu()
    predict_index = output.data.numpy().argmax()
    #print('Prediction: ',class_names[predict_index],'\n')
    return class_names[predict_index],image_datasets['train'].classes[predict_index]


# In[87]:


from glob import glob
from PIL import Image

test_img_paths = sorted(glob('/data/dog_images/test/*/*'))
# Shuffle the list and display first few rows
np.random.shuffle(test_img_paths)
test_img_paths[1:9]

for img_path in test_img_paths[0:6]:
    predict_breed_transfer(img_path)





import random
import matplotlib.image as mpimg
def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)
    breed,name = predict_breed_transfer(img_path)
        
    # display test image
    fig = plt.figure(figsize=(16,4))
    
    if(is_human):
        print('Hi human')
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')

        # display sample of matching breed images
        subdir = '/'.join(['/data/dog_images/valid', str(name)])
        file = random.choice(os.listdir(subdir))
        path = '/'.join([subdir, file])
        ax = fig.add_subplot(1,2,2)
        img = mpimg.imread(path)
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
        plt.title(breed)
        plt.axis('off')
        plt.show()   
        print("You look like ..." + breed)
        print("\n"*3)
        return
    
    elif(is_dog):
        print("Hi Dog")
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')

        # display sample of matching breed images
        subdir = '/'.join(['/data/dog_images/valid', str(name)])
        file = random.choice(os.listdir(subdir))
        path = '/'.join([subdir, file])
        ax = fig.add_subplot(1,2,2)
        img = mpimg.imread(path)
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
        plt.title(breed)
        plt.axis('off')
        plt.show()   
        print("You look like ... " + breed)
        print("\n"*3)
        return
    
    else:
        print('Error')
        ax = fig.add_subplot(1,2,1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        plt.axis('off')
        plt.show()    
        print("\n"*3)
        return




## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)






