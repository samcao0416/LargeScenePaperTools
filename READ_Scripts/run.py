import os
import argparse
import yaml

def arg_parse():
    parser = argparse.ArgumentParser()
    ## Agi Arguments
    parser.add_argument("-A", "--agi", type = str, required = True, help = "path of metashape executable file")
    parser.add_argument("-V", "--agi_ver", type = str, choices = ['201', '165'], default = '201', help = "version of agi metashape")
    
    parser.add_argument("--img_folder", type = str, required = True)
    parser.add_argument("--xml", type = str, default = None, help = "path of blocks exchange format path")
    parser.add_argument("--workspace", type = str, default = None, help = "path of saving all the files")

    parser.add_argument("--read", type = str, required = True, help = "path of READ folder")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()

    metashape_excecutive_path = args.agi
    metashape_version = args.agi_ver
    img_folder = args.img_folder
    xml_file_path = args.xml
    workspace_path = args.workspace
    if workspace_path is None:
        workspace_path = os.path.join(os.path.dirname(img_folder), os.path.basename(img_folder) + "_READ_workspace")
        os.makedirs(workspace_path, exist_ok=True)
    read_folder = args.read

    os.system("%s -r ../AgiScripts/agi_read_%s.py --img_folder %s --xml %s --workspace %s" %(metashape_excecutive_path, metashape_version, img_folder, xml_file_path, workspace_path))
    