import cv2
import torch
from model import Net
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODEL_NAME = "./model.pth"
model = Net()
model.load_state_dict(torch.load(SAVED_MODEL_NAME))
model=model.to(device)
model.eval()

def img_to_tensor(img):
    img = Image.open(img).convert('L')
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: 1 - x)])
    img = transform(img)
    return img

def main():
    with torch.no_grad():
        img_tensor=img_to_tensor('./infer.jpg').unsqueeze(0)
        img_tensor=img_tensor.to(device)
        output = model(img_tensor)
        softmax =torch.nn.Softmax(dim=1)
        output=softmax(output)
        pred = torch.argmax(output, 1).item()
        conf = output[0][pred].item()
        x=round(float(conf),2)


        image=cv2.imread('./infer.jpg')
        font=cv2.FONT_HERSHEY_SIMPLEX
        org=(20,30)
        fontScale=0.5
        Color=(0,0,0)
        thickness=2
        image=cv2.putText(image,f'pred:{pred},conf:{x*100:.2f}%',org,font,fontScale,Color,thickness)
        cv2.imwrite('./infer11.jpg',image)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


#torch.nn.Softmax(dim=1) 会把 Logits 转换成0-1 之间的概率分布（和为 1）
#因为mnist数据库中的照片都是黑底白字的，所以需要翻转transforms.Lambda(lambda x: 1 - x)


