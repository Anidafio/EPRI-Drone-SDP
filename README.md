# Automated Drone-based T&D Asset Inspection and Routing (EPRI_ASSET)

## Introduction

The objective of the EPRI_ASSET project is to create a system consisting of an Unmanned Aircraft System (UAS) and accompanying hardware and software such as a camera and computer vision and AI model. The UAS will be used for inspection of desired Transmission and Distribution (T&D) systems focusing on pole detection and identification using longitude and latitude values.

Using a DJI Mini 3 Pro, the process for this will be as follow: Gather drone pictures -> Take in the pictures and extract the EXIF data -> Run the images through the object detection model -> Run the images through the depth estimation model -> Translate the GPS coordinates from the drone to the desired power poles -> Upload the data to ArcGIS for display.

The object detection model being used for this is a custom [Yolov8](https://github.com/ultralytics/ultralytics) model trained for detecting power poles.

The depth estimation model being used for this is the [Zoe Depth](https://github.com/isl-org/ZoeDepth) NK model.

## Installing and Usage

To clone this repo `git clone https://github.com/mdangc/EPRI-Drone-GIS`

Required libraries and suggested versions using Python 3.10.14:

| Library       | Version |
|:--------------|--------:| 
| pytorch       | 2.3.0   |
| torchvision   | 0.18    |
| timm          | 0.6.7   |
| geopy         | 2.4.1   |
| ultralytics   | 8.1.14  |
| piexif        | 1.1.3   |

A python environment that can run the program can be installed by doing `conda env create -f environment.yml` in folder where the repository was cloned.

The models required for this program are automatically downloaded as needed in the program. However, if you want source them yourselves, below are the links for them:

Custom [Yolov8](https://github.com/ultralytics/ultralytics) Object Detection model.

Pre-existing [Zoe Depth](https://github.com/isl-org/ZoeDepth) Depth Estimation NK model.

After cloning and moving into the cloned folder, the program can be run using `python main.py` or your perfered way to run python programs.

There are a few arguments in the gui once the program has started. These include:

### Directories

_Images Directory_: This is where the pictures from the drone that are wanted to analyzed are placed.

_JSON Directory_: This is where the JSON files created from extracting the image metadata are to be placed.

_Object Detection Directory_: This is where the output images with bounding boxes from the object detection are to be placed.

_Depth Estimation Directory_: This is where the output images from the depth estimation are to be placed.

The only directory that is required to be filled in is the image directory. If any of the other directories is not filled a default directory.

### Checkboxes

_Use Kalman Filter?_: This enables the Kalman filter when translating the GPS coordinates from the drone to the power poles. Default is using the raw GPS data from the drone.

_Crop Images?_: This crops the object detection images around the pole bounding boxes.

### Buttons

_Detect Objects_: This runs the object detection model on the images found in the images directory and outputs in the object detection directory. 

_Extract Metadata_: This uses the [EXIFtool](https://github.com/exiftool/exiftool) to extract the metadata from the images in the images directory and outputs it into respective JSON files in the JSON directory.

_Depth Estimation_: This uses the depth estimation model on the images in the images directory and outputs the results in the depth estimation directory.

_Translate GPS_: This translates the GPS coordinates from the drone to the poles. This requires all three options above (extract metadata,  detect objects, and depth estimation) to be run for it to work properly.

## Drones and SDKs
Different DJI drones were reasearched for the capabilities. The researched drones include the DKI Drone FPV, DJI Phantom 4 RTK, DJI Mavic Air 2, and DJI Mini 3 Pro. The team also researched [DJI SDK](https://developer.dji.com/). The SDK includes a Mobile SDK that allows for the use of mobile applications, the Payload SDK that allows for additional paylods on the drone such as LiDAR sensors, Windows SDK for using Windows applications, and the Onboard SDK for running programs using the drone's onboard capabilities.

## Data Acquisition

For the training of the models, 30,000+ training images of power poles and related equipment was given from EPRI through the use of kaggle and a shared box drive. Additional test images and data was given to the team through the box drive as the project went on to test the object detection, depth estimation, and GPS translation.

For the use of the program, drone pictures from a drone like the DJI Mini 3 Pro (only tested drone, others may be used but results may vary) of power poles are required. These drone pictures must have the drone data embedded in the pictures EXIF data. The program will extract the drone data using a subprocess to the program called [EXIFtool](https://github.com/exiftool/exiftool). Once the drone data is extracted into JSON files, it can be used in the rest of program.

## Reviewed AI Techniques

### Object detection Model

The Yolov8 libraries were used for the object detection model. One of the main advantages of this library was its ease of use and widespread use. Other object detection models required much more time to write code and had much less information about possible errors for debugging already at the model training stages.

Yolov8 provides a large number of flexible settings, which made it possible to run model training on devices with different computing power, this made it possible to quickly experiment with different model settings to find the most optimal and effective ones.

## Data Analysis Approaches

There are a few ways to gather the data from the DJI Mini 3 Pro for use in the program. The first is through extracting the EXIF data from the drone pictures and the second is to use the .srt subtitle data that is given when a video is taken on the drone. Both give GPS data however, the .srt files do not give the rest of the pertinate data such as flight speed (for the kalman filter), or camera and drone data such a pitch, yaw, and roll. Because of this, the EXIF data is used. This limits the process to only gathered pictures from the drone.

To extract this data, the [EXIFtool](https://github.com/exiftool/exiftool) is used. This is called using a subprocess in the python program and this the extracted data is saved into JSON files for use in the kalman filter and the ArcGIS processes.

The data is used in the kalman filter in attempt to predict/estimate more accurately the GPS coordinates of the drone compared to the raw GPS readings from the drone. This GPS data and the IMU data is used in this. The data is also used to translate the GPS coordinates to the power poles that are detected in the object detection process. The GPS latitude, longitude, depth estimation values and bearing degree (yaw) are used for this translation. This is then given to the ArcGIS process where the data is visualized on a map showing the FOV of the camera, and uses the pitch, yaw, roll, and other data from the drone the accurately display the images on the map.

## Model Development

### Object detection Model:

As a dataset for training the model, about 30,000 images were provided on which 6 classes of objects were marked: (crossarm, cutouts, insulator, pole, transformers, background_structure)
When creating an object detection model, experiments were carried out with various settings. One of the counter-intuitive effects turned out to be that the model works better if it is trained on a larger number of classes. Thus, 2 identical models for detecting electric poles produced different efficiencies if they were trained for 2 classes or 6, in favor of the latter.
In other settings, their natural increase led to better results. Higher resolution, more epochs increased efficiency.

The final model processes photos with a vertical resolution of 1280 pixels (automatically stretching or compressing the input image) and spends about 150 milliseconds per image. (test conducted on nvidea rtx 2060 video card)

### Depth Estimation Model:

ZoeDepth works by estimating metric depth from a single image. It involves preloading existing models and processing images to predict depth. The framework allows for using models directly via torch hub or loading them manually from a local copy. 
It supports different models for various depth estimation tasks, and users can make predictions on local files, tensors, or images from URLs

For example, the NYU-Depth V2 and KITTI datasets — the NYU-Depth V2 is an indoor dataset while the KITTI is an outdoor dataset, the former will have a much lower range of depth values as compared to the latter.

At the moment, we are facing the challenge of training ZoeDepth on our customer dataset. In EPRI's images, we are missing the ground truth data which was already included in KITTI and NYU dataset. 

#### Parameters in ZoeDepth:
In configuration, we set ZoeDepth from "infer" to "eval" mode to improve the accuracy. 
```
# ZoeD_N
conf = get_config("zoedepth", "eval")
model_zoe_n = build_model(conf)

# ZoeD_K
conf = get_config("zoedepth", "eval", config_version="kitti")
model_zoe_k = build_model(conf)

# ZoeD_NK
conf = get_config("zoedepth_nk", "eval")
model_zoe_nk = build_model(conf)
```
## Data and Model Analysis Results

### Object detection Model:

During training of our object detection model, we divided our dataset about 17/83 percent proportion for validation and training. After the 50th epoch, that's how long we trained our model, we achieved following results:  
box loss: 0.75457  
class loss: 0.34418  
mAP(50): 0.85466  
mAP(50-95): 0.69855  
mAP(50) is mean average precision at an intersection over union (IoU) threshold of 0.50  
mAP(50-95) is the average of mean average precision at varying IoU thresholds, ranging from 0.50 to 0.95  
That is results for validation of our 6 class object detection model. Even though we need to detect only power poles, we chose to train model for all 6 classes, because it performed much better rather than 1 or 2 class models, so actual results for poles only should be better.

Also we tested our model on pictures that were not in the dataset, and it was able to detect almost every power pole. There was some garbage detection of poles that was not actually on picture, but their confidence rate was very low, so it is easy to ignore. One of the interesting quirks of our model is that it detects poles better if pictures were made from the drone, rather from the ground. It can be explained by our train dataset, which was 100% made from drones. 

![Image of detected pole with garbage detection](https://github.com/mdangc/EPRI-Drone-GIS/blob/ld_combined/doc_images/EPRI_Distro10_20210421%20(6).JPG)

## Mapping
The final step of this project is to output all the data processed and given by the python program in a Geographic Information System (GIS) like [ArcGIS](https://www.arcgis.com/index.html) or [QGIS](https://www.qgis.org/en/site/). Both options were researched and considered when starting the project. ArcGIS is the industry standard for GIS and has a multitude of libraries and projects that can be used and implemented based on one's needs. It also has in depth documentation and fourms that can be used if there are any issues. QGIS is a free and open source option to ArcGIS and has many of the same basic features. However, in testing, ArcGIS was chosen as it had more robust features and was much easier to work with.

The mapping phase of our project employs ArcGIS Pro, a GIS software, to create an interactive map of power poles. Users can click on any power pole on the map to access related data, such as GPS coordinates, altitude, and associated images, videos, and sensor data. Initially, our project focused on geotagging with the dataset provided by EPRI. This crucial first step involved tagging images with geographic coordinates, leveraging the inherent geolocation data in image formats like JPG or PNG. ArcGIS Pro efficiently extracts this information, facilitating user navigation through the image catalog and enhancing dataset management.

Data processing and image cataloging are vital because precise GPS coordinates integrate the datasets into the GIS system, ensuring accurate placement on the map. This accuracy is essential for analyzing spatial relationships and temporal patterns, such as changes in landscape, infrastructure development, or environmental degradation. Furthermore, drone flights typically generate a large volume of imagery. Efficient cataloging within the GIS platform aids in organizing, searching, and swiftly retrieving specific images, enhancing project efficiency.

Recognizing the importance of image cataloging, we implemented a management system known as the Oriented Imagery Catalog (OIC) within ArcGIS Pro. OIC enables users to manage, explore, and view imagery that is not oriented vertically, providing tailored solutions for our project needs.

Additionally, ArcGIS Pro features "QuickCapture," a tool that simplifies data uploading via mobile devices, such as iOS and Android. QuickCapture allows users to capture data with a single click, ideal for operators who need to document observations quickly without navigating complex online GIS interfaces. Data captured during drone inspections can be shared in real-time with team members or stakeholders. QuickCapture also offers a customizable user experience, designed specifically for our project requirements.

Within our existing GIS database, we leverage QuickCapture's integration for automatic geotagging and integration of captured images. For example, in our "Onsite Photos" catalog, users can take pictures of the poles, which are then automatically uploaded to the ArcGIS Online catalog under the "content" section.

## Hurdles
Throughout working on this project, the team had to overcome some hurdles. This section will describe these hurdles and how we overcame them.

First, implementing windows support for our program proved more challenging than initially thought. Some of this comes from the tool used for extracting EXIF data as well as libraries such as the python libraries needed for ArcGIS requiring additional effort to get working. For getting EXIF tool working on windows, the windows executable was grabbed and a check in the python file was implemented to check which os the user is on. Depending on the OS the user is using, the corresponding EXIF tool and commands are used. The output between the OSes is identical. The issue with the ArcGIS python libraries also fixed itself as we got further into our development. This is believed to because of an update to the libraries.

Secondly, on both windows and linux, Pytorch threw errors when trying to use it due to a issue with the default install of the timm library. It was found installing specifically the 0.6.7 version of timm fixes this. This can be done by running `conda install timm=.0.6.7`. Alternatively, you can do `conda env create -f environment.txt` in folder where the repository was cloned to clone the environment used to create this.

## Further Improvements

Due to the limitations of the DJI Mini 3 Pro, the current solution is only able to use images from the drone. This is because the way the DJI Mini 3 Pro processes data of pictures versus video. The drone only give GPS data with video, but gives GPS, camera data, and all other drone data with pictures. For this reason, the prototype ONLY works with pictures. It is wanted to add video support but there is no plan to further it.

Due to time constraints, the team was not able to spend time to get the prototype working real time on the hardware on the drone. This is entirely possible with the DJI SDK using the onboard controller of the drone or a computer connected live to the drone feeding data back and forth.

## Conclusion
The EPRI_ASSET Team spent their senior year (23-24 school year) working on this project. There were many ups and downs in the process. The team ran into issues early with video support as well as the kalman filter as well as issues with the object detection not training as well as initially hoped for. The team was able to figure out early issues with extracting the data from the drone and parsing it to a readable format as well as issues with the depth estimation and get both working in record time. Howeve regardless of this, the team found success in the project and ultimately finished with a working product. 

The EPRI_ASSET Team would like to thank UNC Charlotte and EPRI for this amazing experience and the opportunity to work together and build this project. This experience was invaluable.
