from imutils import paths
import pickle
import cv2
import os
import face_recognition
imagePaths = list(paths.list_images("Dataset"))

# khởi tạo list chứa known encodings và known names (để các test images so sánh)
# chứa encodings và tên của các images trong dataset
knownEncodings = []
knownNames = []

# duyệt qua các image paths
for (i, imagePath) in enumerate(imagePaths):
    # lấy tên người từ imagepath
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load image bằng OpenCV và chuyển từ BGR to RGB (dlib cần)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    boxes = face_recognition.face_locations(rgb, model="cnn")    

   
    encodings = face_recognition.face_encodings(rgb, boxes)  

    
    # Lý tưởng nhất mỗi ảnh có 1 face và có 1 encoding thôi
    for encoding in encodings:
        # lưu encoding và name vào lists bên trên
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump (lưu) the facial encodings + names vào ổ cứng
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))