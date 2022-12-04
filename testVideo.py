import os, cv2
import os, torchvision, torch, cv2, glob
import numpy as np
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn
# Flask utils
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from skimage.io import imsave
from os.path import isfile, join
MODEL_PATH = 'model_office.pth'
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    model.to(device)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    cudnn.benchmark= True
    if os.path.exists(MODEL_PATH):
        temp = torch.load(MODEL_PATH)
        model.load_state_dict(temp['state_dict'])
    return model
model = load_model()
def file_loader_for_testing_segmentation(file_name):
    img = Image.open(file_name)
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    val_transformation = A.Compose([
        A.Resize(height=256, width=256),
        ToTensor(normalize=imagenet_stats)
            ])  
    if val_transformation:
        img = val_transformation(**{"image": np.array(img)})["image"]
    img = img.to('cuda')
    temp = img.size()
    img = img.reshape(1, temp[0], temp[1], temp[2])

    return img


def convert_frames_to_video(pathIn,pathOut,fps,original):
    cap = cv2.VideoCapture(original)
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + '/' + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
        # 0x00000021
        # cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()



def videoFrames(path, name):
    vidcap = cv2.VideoCapture(path)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            if not os.path.exists("testing/segmentation/frames/" + name):
              os.mkdir("testing/segmentation/frames/" + name)
            cv2.imwrite("testing/segmentation/frames/" + str(name) + "/image" +str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 1
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    pathToFrames = "testing/segmentation/frames/" + str(name) + '/'
    return pathToFrames

def model_predict_video(allPathToFrames, frame, pred_path):
    for i in allPathToFrames:
        test_file_input = file_loader_for_testing_segmentation(frame + '/' + i)
        with torch.no_grad():
            preds = torch.sigmoid(model(test_file_input))
            preds = (preds > 0.5).float()
            mask_pred = pred_path + i
            torchvision.utils.save_image(preds, mask_pred)

def predictVideo(video_path):
    name = video_path.split('/')[-1].split('.')[0]
    
    pathToFrames = videoFrames(video_path, name)
    pred_path = "testing/segmentation/framesPred/" + name + '/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    allPathToFrames = os.listdir(pathToFrames)
    frame = os.path.join('testing/segmentation/frames/', name)
    model_predict_video(allPathToFrames, frame, pred_path)
    
    frames_path = 'testing/segmentation/frames/' + name
    frames_pred_path = 'testing/segmentation/framesPred/' + name
    masks_imposed_path = 'testing/segmentation/masksImposed/' + name
    if not os.path.exists(masks_imposed_path):
        os.mkdir(masks_imposed_path)
    for i in os.listdir(frames_path):
        original = frames_path + '/' + i
        seg = frames_pred_path + '/' + i 
        mask_imp = masks_imposed_path + '/' + i
        
        original_image = cv2.imread(original)
        
        original_image = cv2.resize(original_image, (256, 256),interpolation = cv2.INTER_AREA)
        
        seg_image = cv2.imread(seg)
        added_image = cv2.addWeighted(original_image,0.99,seg_image,0.99,0)
        cv2.imwrite(mask_imp, added_image)
        
    path_in = 'testing/segmentation/masksImposed/' + name
    path_out = 'testing/segmentation/predVideos/' + name + '/video.mp4'
    if not os.path.exists('testing/segmentation/predVideos/' + name):
        os.mkdir('testing/segmentation/predVideos/' + name)
    convert_frames_to_video(path_in, path_out, 1, video_path)
    
predictVideo('vecteezy_white-old-building-wall-with-a-crack_1623873.mp4')