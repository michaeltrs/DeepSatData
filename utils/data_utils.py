import re
import os
import zipfile

# dir_name = 'C:\\SomeDirectory'
# extension = ".zip"
#
# os.chdir(dir_name) # change directory from working dir to dir with files

def unzip_all(dir_name, extension=".zip"):
    for item in os.listdir(dir_name):
        if item.endswith(extension):
            file_name = os.path.join(dir_name, item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(dir_name)
            zip_ref.close()
            os.remove(file_name)


def find_number(text, c, single=True):
    val = re.findall(r'%s(\d+)' % c, text)
    if single:
        val = val[0]
    return val
