import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np

transforms_test = transforms.Compose([
    transforms.CenterCrop((1500, 800)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image =Image.open(image_name).convert('RGB')
    image = transforms_test(image)
#     image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image #.to(device)  #assumes that you're using GPU

## 데이타 체크
import torchvision
#import matplotlib.pyplot as plt
#def imshow(inp, title=None):
#    """Imshow for Tensor."""
#    inp = inp.numpy().transpose((1, 2, 0))
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    inp = std * inp + mean
#    inp = np.clip(inp, 0, 1)
#    plt.imshow(inp)
#    if title is not None:
#        plt.title(title)
#    plt.pause(0.001)  # pause a bit so that plots are updated

def onehot_label(row):
    p = re.compile('[A-Z]+')
    multi_label = row.split('/')[3].split('_')[1]
    multi_label = p.findall(multi_label)
    multi_label_list = torch.LongTensor([ord(alpha)-65 for alpha in multi_label[0]])
    y_onehot = torch.nn.functional.one_hot(multi_label_list, num_classes=10)
    y_onehot = np.array(y_onehot.sum(dim=0).float())
    return y_onehot

def onehot2abc(row):
    if type(row) == str:
        row= onehot_label(row)
    argmax_= np.where(row==1)
    return list(map(chr, [x+65 for x in argmax_[0].tolist()] ))

@st.cache
def load_model():
    from efficientnet_pytorch import EfficientNet

    model_name = 'efficientnet-b3'
    save_model_name = 'eff-b3'
    num_classes = 10
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

    #weights_path = 'best_model_eff3_v=t.pt'
    weights_path = 'best_model_cls_all.pt'
    state_dict = torch.load(weights_path, map_location='cpu') # , map_location=device)  # load weight
    #model.load(state_dict)
    model.load_state_dict(state_dict, strict=False)  # insert weight to model structure
    #model = model.to(device)
    return model


@st.cache
def load_reg_model():
    from efficientnet_pytorch import EfficientNet

    reg_model_name = 'efficientnet-b3'  
    reg_save_model_name = 'eff-b3'
    num_classes = 1
    reg_model = EfficientNet.from_pretrained(reg_model_name, num_classes=num_classes)


    #reg_weights_path = 'best_model_reg_3.pt'
    reg_weights_path = 'best_model_reg_all.pt'
    state_dict = torch.load(reg_weights_path, map_location='cpu')  # load weight
    reg_model.load_state_dict(state_dict, strict=False)  # insert weight to model structure


#     reg_model = reg_model.to(device)
    return reg_model



def model_inference(image_filename):
    image = image_loader(image_filename)
    #imshow(image.cpu().squeeze())
    model.eval()
    reg_model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs = image #.to(device)
        outputs = model(inputs)
        preds = [1 if x > 0.5 else 0 for x in
                 outputs.squeeze().tolist()]  # the class with the highest energy is what we choose as prediction
        reg_outputs = reg_model(inputs)
        predicted = torch.round(reg_outputs).int()
        predicted = predicted.squeeze()
        
    return onehot2abc(np.array(preds)), predicted


# GradCAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image


import torch.nn.functional as F

class Multi_GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        scores = []
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        
#             score = logit[:, class_idx].squeeze()
        elif type(class_idx)== list:
            for idx in class_idx:
                s = logit[:, idx].squeeze()
                scores.append(s)
        for i, score in enumerate(scores):
            self.model_arch.zero_grad()
            score.backward(retain_graph=retain_graph)
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()

            alpha = gradients.view(b, k, -1).mean(2)
            # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
            weights = alpha.view(b, k, 1, 1)


            saliency_map = (weights*activations).sum(1, keepdim=True)

            saliency_map = F.relu(saliency_map)

        
            saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if i == 0:
                saliency_map2 = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            else:
                saliency_map2 += (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

                    
        return saliency_map2, logit

    def __call__(self, input, class_idx=None, retain_graph=True):
        return self.forward(input, class_idx, retain_graph)

def inference(image_filename):
    image = image_loader(image_filename)
    #imshow(image.cpu().squeeze())
    model.eval()
    reg_model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        inputs = image #.to(device)
        outputs = model(inputs)
        preds = [1 if x > 0.5 else 0 for x in
                 outputs.squeeze().tolist()]  # the class with the highest energy is what we choose as prediction
        
        reg_outputs = reg_model(inputs)
        predicted = torch.round(reg_outputs).int()
        predicted = predicted.squeeze()

    masks = []
    heatmaps = []
    results = []    
    images = []

    
    pred_label = [i-65 for i in list(map(ord, onehot2abc(np.array(preds))))] 
    num_label = len(pred_label)
    for i in pred_label:
        mask, _ = gradcam(inputs, class_idx = i)
        heatmap, result = visualize_cam(mask, image.cpu().squeeze())
        masks.append(mask)
        heatmaps.append(heatmap)
        results.append(result)
    mask, _ = multi_gradcam(inputs, class_idx = pred_label)
    heatmap, result = visualize_cam(mask, image.cpu().squeeze())
    masks.append(mask)
    heatmaps.append(heatmap)
    results.append(result)
    
    images.extend([image.cpu().squeeze()] + results)
    grid_image = make_grid(images, nrow=num_label+2)
    
    return onehot2abc(np.array(preds)) , grid_image, predicted


    
    
st.title('블록 패턴 추출')
model = load_model()
reg_model = load_reg_model()


target_layer = model._blocks[25]
gradcam = GradCAM(model, target_layer)
multi_gradcam = Multi_GradCAM(model, target_layer)

st.markdown('📸 카메라로 직접 블록구조를 촬영하실 수 있습니다.')

picture = st.camera_input("Take a picture")

uploaded_file = st.file_uploader("Choose an image...")

option = st.selectbox(
     '샘플 사진으로 테스트해보세요.',
     ('AF', 'ABCG', 'CDEFI'))

st.write('You selected:', option)

if option =='AF':
    uploaded_file = 'sample/02_AF_N05_02.JPG'
if option =='ABCG':
    uploaded_file = 'sample/04_ABCG_N11_11.JPG'
if option =='CDEFI':
    uploaded_file = 'sample/05_CDEFI_N17_04.JPG'

if picture:
    uploaded_file = picture

if uploaded_file is not None:
    # src_image = load_image(uploaded_file)

    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    # st.write(os.listdir())


    answer, grid_image, count_block = inference(uploaded_file)

    st.write(f"패턴: {answer}, 블럭 개수: {count_block}개")

    if st.button("GradCAM"):
        st.image(np.transpose(grid_image.numpy(), (1,2,0)), clamp=True)







