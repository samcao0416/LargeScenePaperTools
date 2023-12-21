# Version 2.0.1

## class Metashape.Camera
camera.Type: 'Regular' or 'KeyFrame'
camera.calibration.load(path, format=CalibrationFormatXML)
camera.calibration.save(path, format=CalibrationFormatXML)
camera.calibration.sensor.type: Metashap.sensor.type
camera.sensor.type: Metashape.Sensor.Type.Frame / Fisheye / Spherical / Cylindical / RPC
camera.type: Metashape.Camera.Type.Regular or Metashape.Camera.Type.KeyFrame

To import the calibration to the sensor, you need to use the following code:
```
calib = Metashape.Calibration()
calib.load(path)
sensor.user_calib = calib
calib.fixed = True
```