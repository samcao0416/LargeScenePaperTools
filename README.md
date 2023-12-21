# LargeScenePaperTools
Scripts for Large Scene Dataset and Benchmark Paper

## Folder Structure
### AgiScripts
* **agi_201.py**
This is the running scripts for Agi Metashape Pro 2.0.1 Version.
The official Document is on [Metashape Python Reference 2.0.1](https://www.agisoft.com/pdf/metashape_python_api_2_0_1.pdf).
**The scipts did the following steps:**
    1. Create a new project and a new chunk
    2. Add the .jpg / .png images in the given folder path
    3. import the blocks exchange format camera calibrations
    4. export the reference txt file and import it into the same chunk, so as to add the reference information.
        n: image label
        xyz: coordinates
        abc: rotation angles
        uvw: estimated coordinates
        def: estimated orientation angles
    5. Modify xyz error to 0.03 meter.
    6. Align the images
    7. Build depth images and pointcloud
    8. Export block exchange format camera calibrations and pointcloud
      
    **How to use the script**
    ```
    cd AgiScripts
    <Metashape Folder>/Metashape.exe -r agi_201.py \ 
                                    --img_folder <Image Folder Path> \
                                    --xml <Blocks Exchange Format xml Path> \
                                    --workspace <Workspace folder>
    ```

    **Notes:**
    This scrips only fits Metashape 2.0.1. For other version, you may check the other version's document and modify the scripts.  
    e.g. For Metashape 1.6.5, you may check [Metashape Python Reference 1.6.5](https://www.agisoft.com/pdf/metashape_python_api_1_6_5.pdf)  
    buildPointCloud <-> buildDenseCloud
