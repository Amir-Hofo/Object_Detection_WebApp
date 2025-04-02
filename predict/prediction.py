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
    image= np.array(image)
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2= map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 2)
        cv2.putText(image, f"{label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    return image

def prediction_fn(image, model, min_score= 0.65):
    boxes, labels, scores= object_detection(image, model, min_score)
    image= visualization(image, boxes, labels, scores)
    return image