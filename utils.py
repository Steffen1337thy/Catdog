#Some helpers

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd




def open_protocol(project_name):
    document_name = project_name + ".xlsx"
    try:
        protocol = pd.read_excel(document_name)
        current_model_no = str(protocol['model_no'].iloc[-1] + 1)

    except:
        protocol = pd.DataFrame(columns=['model_no','train_size','val_size','test_size' ,'epochs', 'batch_size',
                                 'learning_rate', 'momentum', 'test_acc', 'graphic'])
        writer = pd.ExcelWriter(document_name, engine='xlsxwriter')
        protocol.to_excel(writer, sheet_name='protocol', index = False)
        writer.save()
        current_model_no = "1"
    return document_name, current_model_no

def save_protocol(document_name, current_model_no, train_size, val_size, test_size, epochs, batch_size, learning_rate,
                  momentum, test_acc, graphic):
    protocol = pd.read_excel(document_name)
    graphic = graphic + '.jpg'
    protocol.loc[protocol['model_no'].max()] = [current_model_no, train_size, val_size, test_size, epochs, batch_size,
                                         learning_rate, momentum, test_acc, graphic]
    writer = pd.ExcelWriter(document_name, engine='xlsxwriter')
    protocol.to_excel(writer, sheet_name='protocol', index=False)
    writer.save()


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    predictions = predictions.cpu()
    y = y.cpu()
    y1= y.numpy()[:,:1].reshape(64)
    predictions1 = 1- predictions.numpy()
    equal = np.equal(predictions1, y1)
    mean = np.mean(equal)
    return mean
    #return np.mean(np.equal(predictions.numpy(), y.numpy()[:,:1].reshape(100,1)))

def create_plot(train_lossd, val_lossd, train_accd, val_accd, epochs, fig_name):

    fig,(ax0, ax1) = plt.subplots(2)

    fig.dpi = 300

    ax0.plot(*zip(*train_accd.items()), label = 'training accuracy')
    ax0.plot(*zip(*val_accd.items()), label = 'validation accuracy')
    ax0.legend()
    ax0.set_title(fig_name)


    ax1.plot(*zip(*train_lossd.items()), label= 'training loss')
    ax1.plot(*zip(*val_lossd.items()), label = 'validation loss')
    ax1.legend()

    ax1.set_xlabel("Epoch")
    ax1.set_xticks(np.arange(0, epochs + 1, 5))

    for ax in fig.get_axes():
        ax.label_outer()

    fig_name='figures/' + fig_name + '.jpg'

    fig.savefig(fig_name)
    plt.show()


def train(train_data, val_data, model, epochs, lr, momentum, fig_name):
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

    train_lossd = {}
    val_lossd = {}
    train_accd = {}
    val_accd = {}
    model_name=fig_name + '.pth'


    for epoch in range(1, epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Training
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
        train_lossd[epoch] = loss
        train_accd[epoch] = acc


        # Validation
        val_loss, val_acc = run_epoch(val_data, model.eval(), optimizer)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        val_lossd[epoch] = val_loss
        val_accd[epoch] = val_acc

        # Save Model
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, model_name)
        #print("Current learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])

    create_plot(train_lossd, val_lossd,train_accd,val_accd, epochs=epochs, fig_name=fig_name)
    return val_acc


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for x, y in (tqdm(data)):
        x = Variable(x.cuda())
        y = torch.Tensor(y).cuda()
        y = Variable(y)
        # Get output predictions
        out = model(x)
        model = model.cuda()

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        predictions = Variable(predictions.cuda())
        batch_accuracies.append(compute_accuracy(predictions, y))
        # Compute loss
        loss = F.binary_cross_entropy(out, y)
        losses.append(loss.data.item())

        mean_loss= sum(losses) / len(losses)
        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #scheduler.step(mean_loss)



    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy

def device():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

def image_show(image,topclass,conf, ax=None, title=None, normalize=True ):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    if topclass == 1:
        title = 'dog ' + str(conf)
        ax.set_title(title)
    if topclass == 0:
        title = 'cat ' + str(conf)
        ax.set_title(title)
    plt.show()
    return ax

def image_show_list(image_list,topclass_list,conf_list, ax=None, title=None, normalize=True ):



    if ax is None:
           fig, axes = plt.subplots(2,4)
    axes = axes.flatten()

    for i,ax in enumerate(axes):

        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_list[i] = std * image_list[i] + mean
            image_list[i] = np.clip(image_list[i], 0, 1)

        axes[i].imshow(image_list[i])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].tick_params(axis='both', length=0)
        axes[i].set_xticklabels('')
        axes[i].set_yticklabels('')
        if topclass_list[i] == 1:
            title = 'dog ' + str(conf_list[i])
            axes[i].set_title(title)
        if topclass_list[i] == 0:
            title = 'cat ' + str(conf_list[i])
            axes[i].set_title(title)
    plt.show()
    return ax

def image_transform(image_path):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image = Image.open(image_path)

    image_tensor = test_transforms(image)
    return image_tensor