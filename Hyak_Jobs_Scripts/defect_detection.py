import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet50
from torchvision.models.detection.rpn import AnchorGenerator
from dgl.data.utils import split_dataset
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score
import random
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')


# function to load images from specified folder path
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image: {img_path} - {e}")
    return images

# Finds class and boundary box data in .xml files
def parse_xml_label(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image size
    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)
    depth = int(size_elem.find('depth').text)

    # Extract object annotations
    annotations = []
    for obj_elem in root.findall('object'):
        name = obj_elem.find('name').text
        bbox_elem = obj_elem.find('bndbox')
        xmin = float(bbox_elem.find('xmin').text)
        ymin = float(bbox_elem.find('ymin').text)
        xmax = float(bbox_elem.find('xmax').text)
        ymax = float(bbox_elem.find('ymax').text)
        annotations.append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })

    return width, height, depth, annotations

# Specify the folder path containing the images
folder_path = './mos2'

# Load images from the folder
original_images = load_images_from_folder(folder_path)

# Specify the directory path containing the XML labels
labels_dir = './data'

# Load .xml labels for all files in the directory
# This is a dictionary object
labels = {}
for filename in os.listdir(labels_dir):
    if filename.endswith('.xml'):
        xml_path = os.path.join(labels_dir, filename)
        width, height, depth, annotations = parse_xml_label(xml_path)
        labels[filename] = {
            'width': width,
            'height': height,
            'depth': depth,
            'annotations': annotations
        }

# This is a class for the data loader for images and labels
class YourDataset(Dataset):
    def __init__(self, images, labels, target_size):
        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),  # resize the image before converting to a tensor
            transforms.ToTensor(),  # then convert to a tensor
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        # Here the index is passed to a list for images but the labels are a dict so the index is 
        # converted to a key
        label_index = index + 1
        key = f"{label_index}.xml"  
        label = labels[key]
    
        # Transforms the image into appropriate sized tensor
        image_tensor = self.transform(image)
        image_tensor = image_tensor.to(device)
    
        # Convert the annotations to tensors
        annotations = label.get('annotations', [])
        annotation_tensors = []
        # This dictionary is for classification of the defects
        class_to_label = {'Point': 0, 'Void': 1, 'Other': 2, 'unlabelled': 0}
        # Here the anotations of the label are turned into tensors
        for annotation in annotations:
            name = annotation.get('name')
            class_label = class_to_label.get(name, class_to_label['unlabelled']) 
            xmin = annotation.get('xmin', 0) / 1024
            ymin = annotation.get('ymin', 0) / 1024
            xmax = annotation.get('xmax', 0) / 1024
            ymax = annotation.get('ymax', 0) / 1024
            
            annotation_tensor = torch.tensor([class_label, xmin, ymin, xmax, ymax])
            annotation_tensors.append(annotation_tensor.to(device))
        # This is done so that the target tensors are the same length
        max_annotations = 171
        while len(annotation_tensors) < max_annotations:
            # Append padding annotation
            # These are weird since the padded boundary box has to be a positive box
            pad_tensor = torch.tensor([0.001, 0.001, 0.002, 0.002, 0.003])  # Padding value
            annotation_tensors.append(pad_tensor.to(device))
        # Stack the bounding boxes into a single tensor for each image
        annotation_tensors = torch.stack(annotation_tensors)  
        return image_tensor, annotation_tensors


# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# Define the ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)) # this will make the output size to be (batch_size, num_channels, 1, 1)
        out = torch.flatten(out, 1) # flatten the tensor starting from dimension 1
        out = self.linear(out)
        return out


# Create the ResNet34
def ResNet34():
    return ResNet(ResidualBlock, [3, 4, 6, 3])

# Create an instance of your Dataset
dataset = YourDataset(original_images, labels, (1024, 1024))

# Load the data from your Dataset
#dataloader = DataLoader(dataset, batch_size=1)
batch_size = 1

train_set, test_set, val_set = split_dataset(dataset, [0.80, 0.1, 0.1], shuffle=True, random_state=21)
print(f'Train Size {len(train_set)}, Test Size {len(test_set)}, Validation Size {len(val_set)}')

train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)



# Instantiate the model
num_classes = 5 # 4 classes + 1 for background
model = ResNet34()
model.fc = nn.Linear(1024, num_classes) # Replace the last fc layer

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.005)

# number of epochs
num_epochs = 50


def get_object_detection_model(num_classes):
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_object_detection_model(num_classes)
model = model.to(device)

training_loss = []
validation_loss = []
# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    print(f"Epoch {epoch}/{num_epochs}")
    model.train()

    for i, (images, annotations) in enumerate(train_loader):
    # Extract boxes and impclass from annotations
        impclass = annotations[:, :, 0]
        boxes = annotations[:, :, 1:]

    # Create targets
    targets = []
    for b, c in zip(boxes, impclass):
        targets.append({
            'boxes': b.to(device),  # Should be a FloatTensor of shape [n_boxes, 4]
            'labels': c.long().to(device),  # Should be a LongTensor of shape [n_boxes]
        })

    # Forward
    loss_dict = model(images, targets)
    # The model returns a dict where the total loss is the sum of all losses
    losses = sum(loss for loss in loss_dict.values())

    # Backward
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    train_loss = losses.detach().item()

    training_loss.append(train_loss)

    print(f'Time Elapsed {(time.time()-start_time) : .2f} Seconds')

print(f'Training Terminated at Epoch {num_epochs}')



model.eval()
validation_loss = []
for epoch in range(num_epochs):
    start_time = time.time()
    print(f"Epoch {epoch}/{num_epochs}")
    model.train()

    for i, (images, annotations) in enumerate(val_loader):
    # Extract boxes and impclass from annotations
        impclass = annotations[:, :, 0]
        boxes = annotations[:, :, 1:]

    # Create targets
    targets = []
    for b, c in zip(boxes, impclass):
        targets.append({
            'boxes': b.to(device),  # Should be a FloatTensor of shape [n_boxes, 4]
            'labels': c.long().to(device),  # Should be a LongTensor of shape [n_boxes]
        })

    # Forward
    loss_dict = model(images, targets)
    # The model returns a dict where the total loss is the sum of all losses
    losses = sum(loss for loss in loss_dict.values())

    # Backward
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    val_loss = losses.detach().item()

    validation_loss.append(val_loss)

    print(f'Time Elapsed {(time.time()-start_time) : .2f} Seconds')

print(f'Validation Terminated at Epoch {num_epochs}')


#Making Loss Plots
fig, axes = plt.subplots(1,2, figsize=(15,5))

axes[0].plot(training_loss, label='Training Loss')
axes[0].set_title('Training Loss vs Epoch')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')

axes[1].plot(validation_loss, label='Validation Loss', color='red')
axes[1].set_title('Validation Loss vs Epoch')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')

plt.suptitle('Loss vs Epoch')

plt.savefig('./LossvsEpoch')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()

plt.savefig('./combinedloss')

image, target = test_set[random.randint(0, len(test_set)-1)]

model.eval()

# Disable computation of gradients
with torch.no_grad():
    # Get the prediction from the model
    prediction = model([image])

def draw_boxes(image, boxes, labels, scores, threshold=0.33):
    # Convert the image from PyTorch tensor to a PIL image and draw the bounding boxes
    image = transforms.ToPILImage()(image.cpu())
    draw = ImageDraw.Draw(image)
    
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=3)
            label_with_precision = f"{label.item()} ({score:.2f})"
            #draw.text((box[0], box[1]), text=str(label.item()))
            draw.text((box[0], box[1]), text=label_with_precision)
    return image

# To draw boxes on the first image
result_image = draw_boxes(image, prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores'])

result_image.save('result_image.jpg')

def combine_tensors(dict_of_tensors, key1, key2):
    tensor1 = dict_of_tensors[key1]
    tensor1 = tensor1.unsqueeze(0)
    tensor1 = tensor1.view(-1,1)
    tensor1 = tensor1.cpu()
    tensor2 = dict_of_tensors[key2]
    tensor2 = tensor2.cpu()

    combined_tensor = np.concatenate((tensor1, tensor2), axis=1)
    return combined_tensor


def calculate_precision(predicted_boxes, ground_truth_boxes, threshold):
    # Convert continuous values to binary labels
    predicted_labels = (predicted_boxes[:, 0] > threshold).astype(int)
    ground_truth_labels = (ground_truth_boxes[:len(predicted_labels), 0] > threshold).type(torch.int)

    # Calculate precision
    precision = precision_score(ground_truth_labels.cpu().numpy(), predicted_labels)
    return precision

print(f"Precision: {calculate_precision(combine_tensors(prediction[0], 'labels', 'boxes'), target.cpu(), 0.33) * 100} % ")
