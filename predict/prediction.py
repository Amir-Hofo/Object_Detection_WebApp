from model import *

def object_detection(image, model, min_score):
    transform= torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image= transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction= model(image)

    boxes= prediction[0]["boxes"]
    labels= prediction[0]["labels"]
    scores= prediction[0]["scores"]

    boxes= boxes[scores >= min_score]
    labels= labels[scores >= min_score]
    scores= scores[scores >= min_score]
    labels= [COCO_CLASSES[label.item()] for label in labels]
    
    return boxes, labels, scores

def visualization(image, boxes, labels, scores):
    image= image.convert("RGB")
    draw= ImageDraw.Draw(image)
    font= ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2= map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline= (0, 255, 255), width= 2)
        text= f"{label} ({score:.2f})"
        draw.text((x1, y1 - 10), text, fill= (0, 255, 255), font= font)
    return image

def prediction_fn(image, model, min_score= 0.65):
    boxes, labels, scores= object_detection(image, model, min_score)
    image= visualization(image, boxes, labels, scores)
    return image