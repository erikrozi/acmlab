import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

MODEL_PATH = "cnn_model.pth"

class ConvolutionalNeuralNet(nn.Module):
  def __init__(self):
    super(ConvolutionalNeuralNet, self).__init__()
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = torch.nn.Dropout(p=0.5)
    self.batchnorm = nn.BatchNorm2d(120)

    self.conv1 = nn.Conv2d(3, 120, 3, padding=1, stride=2)
    self.conv2 = nn.Conv2d(120, 120, 3, padding=1)
    self.conv3 = nn.Conv2d(120, 120, 3, padding=1, stride=2)
    self.conv4 = nn.Conv2d(120, 120, 3, padding=1)
    self.conv5 = nn.Conv2d(120, 120, 3, padding=1,stride=2)
    self.conv6 = nn.Conv2d(120, 120, 5, padding=2)
    self.conv7 = nn.Conv2d(120, 120, 5, padding=2, stride=2)
    self.conv8 = nn.Conv2d(120, 120, 5, padding=2)
    self.conv9 = nn.Conv2d(120, 120, 5, padding=2, stride=2)
    self.conv10 = nn.Conv2d(120, 120, 5, padding=2)
    self.conv11 = nn.Conv2d(120, 120, 5, padding=2, stride=2)
    self.conv12 = nn.Conv2d(120, 120, 5, padding=2)

    # output of x.shape prior to calling fc1
    self.fc1 = nn.Linear(120*1*1, 60)
    self.fc2 = nn.Linear(60, 1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.dropout(x)
    x = self.conv2(x)
    x = self.pool(self.conv3(x))
    x = self.pool(self.conv4(x))
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    x = self.conv10(x)
    x = self.conv11(x)
    x = self.conv12(x)

    x = x.view(-1, 120*1*1) 
    x = self.fc1(x)
    x = self.fc2(x) 
    return x

def predict(image_path):
    model = ConvolutionalNeuralNet()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    image = Image.open(image_path)
    predict_transform = transforms.ToTensor()
    image = predict_transform(image)
    image = torch.unsqueeze(image, 0)
    output = model(image).squeeze().item()
    return output