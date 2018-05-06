

'''
Check if the file content of path is data
data (binary array)
'''
def file_content_is(data, path):
    with open(path, 'rb') as f:
        return data == bytearray(f.read())
