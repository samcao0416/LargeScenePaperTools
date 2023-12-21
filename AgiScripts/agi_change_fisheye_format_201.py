## Version 2.0.1
# This scripts use metashape to change fishaeye agi format to opencv format. 

import Metashape #type: ignore
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-img", "--img_path", type = str, required = True, help = "load at least 1 image to change the format")
parser.add_argument("-xml", "--xml_path", type = str, required = True, help = "path of blocks exchange format path")
parser.add_argument("-out", "--output_path", type = str, default = None, help = "path of saving opencv-format xml file")

args = parser.parse_args()

img_path = args.img_path
xml_path = args.xml_path
output_path = args.output_path

if output_path == None:
    img_folder = os.path.dirname(os.path.dirname(img_path))
    output_path = os.path.join(os.path.dirname(img_folder), os.path.basename(img_folder)+"_READ_workspace", "opencv_format.xml")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. 创建新的项目和chunk
doc = Metashape.Document()
chunk = doc.addChunk()

# 2. 导入一张图像
chunk.addPhotos([img_path])

# 3. 修改相机模型为Fisheye
camera = chunk.cameras[0]
camera.sensor.type = Metashape.Sensor.Type.Fisheye
# camera.calibration.type = Metashape.Sensor.Type.Fisheye
# 4. 读取agisoft xml文件
#camera.calibration.load(path = xml_path, format = Metashape.CalibrationFormatXML)
calib = Metashape.Calibration()
calib.load(path = xml_path, format = Metashape.CalibrationFormatXML)

camera.sensor.user_calib = calib
camera.sensor.fixed = True

# 5. 保存为opencv xml文件
camera.sensor.calibration.save(path = output_path, format = Metashape.CalibrationFormat.CalibrationFormatOpenCV)

print("[ INFO ] Opencv-format xml file saved at %s" %(output_path))