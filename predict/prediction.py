from model import *

def object_detection(image, model, min_score= 0.80):
    ## load image
    transform= torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()])
    image= transform(image).unsqueeze(0)
    ## model 
    with torch.no_grad():
        prediction= model(image)

    boxes= boxes[scores >= min_score]
    labels= labels[scores >= min_score]
    scores= scores[scores >= min_score]
    
    return boxes, labels, scores