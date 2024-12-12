import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataset import load_mask,load_image
from PIL import Image
import os

def draw_contour(image_path,mask_path,save_path):
    image = load_image(image_path)
    plt.imshow(image)
    mask = cv2.imread(mask_path,0)
    # unique_values = np.unique(mask)
    plt.imshow(mask, cmap='gray')
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # 提取轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour_image = np.zeros_like(image)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    plt.imshow(image)

    # 显示图像
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image

def generate_gif_pillow(image_folder, output_path, duration=500, loop=0, ):
    """
    使用Pillow生成动图（GIF）。
    
    参数：
    - image_folder: 存储连续图像的文件夹路径。
    - output_path: 动图保存的路径，例如 'animation.gif'。
    - duration: 每帧的持续时间（毫秒）。
    - loop: 动图的循环次数，0表示无限循环。
    """
    frames = []
    filenames = sorted([fn for fn in os.listdir(image_folder) if fn.endswith(('.png', '.jpg', '.jpeg'))])

    for filename in filenames:
        img_path = os.path.join(image_folder, filename)
        # 使用OpenCV加载图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告：无法加载图像 {img_path}")
            continue
        # 转换为RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 转换为PIL Image对象
        pil_image = Image.fromarray(img_rgb)
        frames.append(pil_image)
    
    if not frames:
        print("没有加载到任何图像，动图未生成。")
        return
    
    # 保存为GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"动图已保存为 {output_path}")

def add_color(image_path,mask_path,save_path):
    # 加载原始图像
    original_image = load_image(image_path)

    # 加载分割掩码
    mask = cv2.imread(mask_path,0)

    # 确保掩码是二值的（0或1）
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # 创建一个彩色掩码（红色）
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 2] = 255  # 红色通道

    # 创建一个半透明的红色层
    alpha = 0.5  # 透明度
    overlay = original_image.copy()
    overlay[binary_mask == 1] = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)[binary_mask == 1]


    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay


if __name__=='__main__':

    # # 示例使用
    # image_folder = 'data/test/instrument_dataset_1/left_frames'          # 替换为您的图像文件夹路径
    # output_gif = 'animation_pillow.gif'  # 替换为您希望保存的动图路径
    # frame_duration_ms = 400         # 每帧持续500毫秒

    # generate_gif_pillow(image_folder, output_gif, frame_duration_ms)
    mask='predictions/UNet/binary/instrument_dataset_1/frame000.png'
    image='data/train/instrument_dataset_1/left_frames/frame000.png'
    save='test.png'
    add_color(image,mask,save)