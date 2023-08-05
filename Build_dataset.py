import cv2 
import os
from mtcnn_align import mtcnn_detect

def dem_so_thu_muc(path):
    count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count
cnn=mtcnn_detect()
video = cv2.VideoCapture(0)
total = 0
id=input("Nhập tên người dùng:")
path="Dataset/"+id+"/"

"""check đã quá giới hạn người nhập hay chưa"""
if dem_so_thu_muc("Dataset")>4:
    print("Đã quá giới hạn người dùng")
    delete=input("Nhập tên người muốn xóa:")
    while not os.path.exists( "Dataset/"+delete+"/"):
        print("thư mục không tồn tại")
        delete=input("Nhập lại tên người muốn xóa:")
    os.rmdir("Dataset/"+delete)
        
"""check đã nhập thông tin hay chưa"""
while  os.path.exists(path):
# if  os.path.exists(path):
    print("Đã nhập thông tin cho người này")
    id_2=input("Nhập tên người dùng:")
    path="Dataset/"+id_2+"/"

os.makedirs(path)
while True:
    ret, img = video.read()
    img = cv2.flip(img,1)
#     new_width = int(img.shape[1] * 0.5)  # Giảm tỷ lệ theo chiều rộng
#     new_height = int(img.shape[0] * 0.5)  # Giảm tỷ lệ theo chiều cao

# # Giảm tỷ lệ frame ảnh
#     scaled_image = cv2.resize(img, (new_width, new_height))
    # cnn.draw_box(scaled_image)
    cv2.imshow("video", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        p = path+str(total)+".jpg"    # điền thêm số 0 bên trái cho đủ 5 kí tự
        image=cnn.crop_face(img)
        cv2.imwrite(p, image)
        total += 1
	# nhấn q để thoát
    elif key == ord("q") or total==15:
	    break


        
print("[INFO] {} face images stored".format(total))
video.release()
cv2.destroyAllWindows()




