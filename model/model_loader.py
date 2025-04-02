from utils import *

class Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.ssd_model= models.detection.ssdlite320_mobilenet_v3_large(
            weights= models.detection.ssdlite.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        
        self.retina_model= models.detection.retinanet_resnet50_fpn(
            weights= models.detection.retinanet.RetinaNet_ResNet50_FPN_Weights.DEFAULT)

        self.faster_rcnn_model= models.detection.fasterrcnn_resnet50_fpn(
            weights= models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        self.fcos_model= models.detection.fcos_resnet50_fpn(
            weights= models.detection.fcos.FCOS_ResNet50_FPN_Weights.DEFAULT)
        
        if model_name== 'faster_rcnn':
            self.model= self.faster_rcnn_model
        elif model_name== 'retina':
            self.model= self.retina_model
        elif model_name== 'ssd':
            self.model= self.ssd_model
        elif model_name== 'fcos':
            self.model= self.fcos_model
        else:
            raise Exception("The entered model name is not correct. \n \
                             Try one of the following options: \n \
                             faster_rcnn \n retina \n ssd \n fcos")
        
    def forward(self, x):
        x= self.model(x)
        return x
    

COCO_CLASSES= [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# COLORS= np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")