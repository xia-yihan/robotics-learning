import cv2
import os

input_path = "images/input.png"

os.makedirs("output", exist_ok=True)

image = cv2.imread(input_path)

if image is None:
    print("图片读取失败，请检查图片路径")
    exit()

height, width, channels = image.shape

print("图像宽度：", width)
print("图像高度：", height)
print("通道数量：", channels)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.rectangle(
    image,
    (50, 50),
    (width - 50, height - 50),
    (0, 255, 0),
    3
)

cv2.putText(
    image,
    "Robot Vision",
    (60, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imwrite("output/gray_image.jpg", gray_image)
cv2.imwrite("output/marked_image.jpg", image)

print("处理完成，结果已保存到 output 文件夹")