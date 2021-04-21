
    # -*- coding: utf-8 -*-

import os
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                # L.append(os.path.join(root, file))
                L.append( file)

                # os.remove(path)
    return L
L=file_name(r'S:\Program Files (x86)\Tencent\QQ\343145366\FileRecv\MIXuav\yolo_uav\JPEGImages')
print(L)
print(len(L))

def remove_file_name(L,file_dir):
    NO_USE=[]

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_names= os.path.splitext(file)[0]
            print(file_names)
            if file_names+'.jpg' not in L:
                useness_file=os.path.join(root, file_names+'.xml')
                NO_USE.append(useness_file)
                os.remove(useness_file)
                # L.append(os.path.join(root, file))
                # L.append( file)
    print(len(NO_USE))
                # os.remove(path)
    return 0
remove_file_name(L,r'S:\Program Files (x86)\Tencent\QQ\343145366\FileRecv\MIXuav\yolo_uav\Annotations')