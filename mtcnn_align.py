import mtcnn
import cv2
class mtcnn_detect:
    def __init__(self) :
        self.test=0
    def crop_face(self,image):
        detector=mtcnn.MTCNN()
        faces=detector.detect_faces(image)
        for face in faces:
            x,y,width,height=face["box"]
        img=image[x:x+width,y:y+height]
        return img
    def draw_box(self,img):
        detector=mtcnn.MTCNN()
        faces=detector.detect_faces(img)
        for face in faces:
            x,y,width,height=face["box"]
        color = (0, 255, 0)  # Đây là màu xanh lá cây
        # Độ dày của đường viền (nếu muốn)
        thickness = 2
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)