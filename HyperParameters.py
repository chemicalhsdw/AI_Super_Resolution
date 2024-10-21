'''
超参数和调用模块
'''
# 文件Real_ESRGAN_master.inference_realesrgan_class 由源代码修改而成，添加了类结构，提升调用效率
from Real_ESRGAN_master.inference_realesrgan_class import *
import cv2
import time
from tqdm import tqdm
'''
通过类实例对象RealESRGANProcessor()调用，输入输出皆为numpy数组
'''

if __name__ == '__main__':
    # 示例使用-单图
    # 为了保证多次调用的速率，请实例化类后 保持此后的多次调用都使用实例方法processer.enhance_image进行超分
    processer = RealESRGANProcessor()


    input_image = cv2.imread('image_01.jpg')

    # 第一次调用会内部初始化，耗时略长
    output_image_first = processer.enhance_image(input_image=input_image)

    # 之后的调用时间约在7~10ms/张
    time1 = time.time()
    output_image = processer.enhance_image(input_image=input_image)
    time2 = time.time()
    print(f'耗时{time2-time1}s')

    # 输出的numpy数组形状为(256, 256, 3) (3通道)
    # print(output_image.shape)

    cv2.imwrite('image_01_out.jpg', output_image)



    # 示例使用-多图超分后组合成视频
    # images_path = r'E:\dek\Haptic artificial intelligence development\6_classification_pre-tasks\save'
    # video_output_path = r'E:\dek\Haptic artificial intelligence development\6_classification_pre-tasks\vedio'
    #
    # # 合成帧为MP4
    # frame_rate = 20  # 视频帧率，可以根据需要调整
    # frame_size = (256, 256)  # 假设所有图像大小相同
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    #
    # # super_resolution_image = Super_Resolution_image()
    # processer = RealESRGANProcessor()
    #
    # for i, cls in enumerate(os.listdir(images_path)):
    #     cls_path = fr"{images_path}\{cls}"
    #
    #     if os.path.isdir(cls_path):
    #         image_files = sorted([f for f in os.listdir(cls_path) if f.endswith('.jpg')])
    #
    #         # 创建视频写入对象
    #         video_path = fr"{video_output_path}\{cls}_new.mp4"
    #         out = cv2.VideoWriter(video_path, fourcc, frame_rate, frame_size)
    #
    #         for image_file in tqdm(image_files):
    #             img_path = fr"{cls_path}\{image_file}"
    #             # 使用OpenCV读取图像
    #             frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 假设图像是灰度图
    #             '''别问为什么，不加这行会出现写入文件失败'''
    #             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #             # 超分辨率处理
    #             # time1 = time.time()
    #             frame = processer.enhance_image(frame)
    #             # time2 = time.time()
    #             # print(f'耗时{time2-time1}s')
    #
    #             # 检查图像是否成功加载
    #             if frame is not None:
    #                 # 写入视频文件
    #                 out.write(frame)
    #
    #                 cv2.imshow('frame', frame)  # 显示帧
    #                 if cv2.waitKey(1) & 0xFF == ord('q'):
    #                     break
    #
    #             else:
    #                 print(f"警告：无法加载图像 {img_path}")
    #
    #         # 释放视频写入对象
    #         out.release()
    #         print(f"类别-{cls}的视频文件已创建")





















