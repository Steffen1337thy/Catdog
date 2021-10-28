from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.nn as nn
from utils import *

def main():
    train_path = 'data/train/'

    # Name of the Project for organisational reasons
    project_name = 'catdog'
    document_name, current_model_no = open_protocol(project_name=project_name)
    fig_name = project_name + '_model_no_' + current_model_no

    #DL - Parameters:
    epochs = 10
    batch_size = 64
    lr = 0.001
    momentum = 0.2
    val_set_relation = 0.1

    # program mode will reduce dataset for quicker execution
    program_mode = False
    # train_mode True for training, False for evaluation on Test set
    train_mode = False


    # this is the model I trained with the best fit.
    # change if you want to test other model or if newer model available.
    model_to_use = 'catdog_model_no_2.pth'


    #Preprocessing Parameters:
    #params seems to be best practice
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])


    # load the data:
    if train_mode:
        # TARGET : [Cat,Dog]
        no_of_batches = int(len(listdir(train_path))/batch_size)
        train_data_list = []
        target_list = []
        train_data = []
        val_data = []
        files = listdir(train_path)
        print('loading ' , len(listdir(train_path)) , ' data in ' , no_of_batches , ' batches' )
        for i in tqdm(range(len(listdir(train_path)))):
            f = random.choice(files)
            files.remove(f)
            img = Image.open(train_path + f)
            img_tensor = transform(img)
            train_data_list.append(img_tensor)
            isCat = 1 if 'cat' in f else 0
            isDog = 1 if 'dog' in f else 0
            target = [isCat, isDog]
            target = target
            target_list.append(target)
            if len(train_data_list) >= batch_size:
                if len(train_data) < no_of_batches * (1-val_set_relation):
                    train_data.append((torch.stack(train_data_list), target_list))
                else:
                    val_data.append((torch.stack(train_data_list), target_list))
                train_data_list = []
                target_list = []
                if program_mode:
                    if len(train_data) == 3:
                        break

        train_size = len(train_data) * batch_size
        val_size = len(val_data) * batch_size
        test_size = None

        print('data load finished')
        print('train set contains: ', len(train_data), ' batches')
        print('val set contains: ', len(val_data), ' batches')

        # I used utils code from another project of mine where I got the test accuracy
        # As we don't have it here, we just set it to None
        # Protocol is something that needs to be updated to better fit on all projects.
        accuracy = None

    # the nn model
    class Netz(nn.Module):
        def __init__(self):
            super(Netz, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.conv_dropout = nn.Dropout2d(0.25)
            self.flatten = nn.Flatten()

            self.fc1 = nn.Linear(64*3*3, 10)
            self.fc2 = nn.Linear(10,2)
            self.relu = nn.ReLU()


        def forward(self, x):
            x = self.layer1(x)
            x = self.conv_dropout(x)
            x = self.layer2(x)
            x = self.conv_dropout(x)
            x = self.layer3(x)
            x = self.conv_dropout(x)
            x = self.flatten(x)

            #print(x.size())
            #exit()
            x = x.view(-1, 64*3*3)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.sigmoid(x)

    model = Netz()
    model.cuda()
    optimizer = optim.Adam(model.parameters())

    if train_mode:
    # train the model and write protocol
        train(train_data, val_data, model, epochs=epochs, lr = lr, momentum=momentum, fig_name=fig_name)

        save_protocol(document_name=document_name, current_model_no=current_model_no,train_size=train_size,val_size=val_size,
                      test_size=test_size,epochs=epochs,batch_size=batch_size,learning_rate=lr,momentum=momentum,test_acc=accuracy, graphic=fig_name)


    #short-cut if you just want to test the model
    else:
        checkpoint = torch.load(model_to_use)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])



    # Move model to CPU
    device = torch.device('cpu')
    model.to(device)

    # Define test directory
    check_on_test = True
    test_dir = 'data/test'
    train_dir = 'data/train'
    image_list = []
    top_class_list = []
    conf_list = []

    for i in range(8):
        # Get random image from test dataset
        classification = 'cat' if random.randint(0,1) == 0 else 'dog'
        if check_on_test:
            image_path = f'{test_dir}/{random.randint(1, 12500)}.jpg'
        else:
            image_path = f'{test_dir}/{classification}.{random.randint(0, 12499)}.jpg'

        # Transform image
        image = image_transform(image_path)

        # Show image
        image_x = image.clone().detach()


        # Set model to evaluation mode
        model.eval()

        # Make prediction
        with torch.no_grad():
            image = image[None, :, :, :]
            ps = torch.exp(model(image))
            top_p, top_class = ps.topk(1, dim=1)
            low_p, low_class = ps.topk(1, dim=1, largest = False)
            conf = top_p.item() / (top_p.item() + low_p.item())

        if top_class == 1:
            print(f'Class: dog (confidence: {conf})')
        else:
            print(f'Class: cat (confidence: {conf})')
        image_x = image_x.numpy().transpose((1, 2, 0))
        image_list.append(image_x)
        top_class_list += top_class
        conf_list.append(str(conf))

    image_show_list(image_list, top_class_list, conf_list)






if __name__ == '__main__':
    main()