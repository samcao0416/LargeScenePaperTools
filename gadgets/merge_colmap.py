import os
import numpy as np

if __name__ == "__main__":

    txt_path_list = [r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\insta_gaussian_image_single_group0\sparse\0\images.txt",
                     r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\insta_gaussian_image_single_group1\sparse\0\images.txt",
                     r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\insta_gaussian_image_single_group2\sparse\0\images.txt",
                     r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\insta_gaussian_image_single_group3\sparse\0\images.txt",
                     r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\insta_gaussian_image_single_group4\sparse\0\images.txt",]

    output_path = r"E:\SkdPaper\underground1010\titan\gaussian_data\block_insta_single\images_all.txt"
    img_name_list = []
    qvec_list = []
    tvec_list = []

    for txt_path in txt_path_list:

        with open(txt_path, "r") as f_in:
            
            while(True):

                line = f_in.readline()

                if not line:
                    break

                line_ele = line.strip()

                if len(line_ele) > 0 and line[0] != "#":

                    elems = line_ele.split()

                    image_id = int(elems[0])
                    
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    camera_id = int(elems[8])
                    image_name = elems[9]
                    try:
                        elems = f_in.readline().split()
                        xys = np.column_stack([tuple(map(float, elems[0::3])),
                                            tuple(map(float, elems[1::3]))])
                        point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    except:
                        xys = None
                        point3D_ids = None

                    if image_name not in img_name_list:
                        img_name_list.append(image_name)
                        qvec_list.append(qvec)
                        tvec_list.append(tvec)

    with open(output_path, "w+") as f_out:
        for idx in range(len(img_name_list)):
            q = qvec_list[idx]
            t = tvec_list[idx]
            image_name = img_name_list[idx]
            output_line = f"{idx +1 } {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {image_name}\n No Content \n"
            f_out.write(output_line)

