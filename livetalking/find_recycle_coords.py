import os

def find_coords():
    for root, dirs, files in os.walk('c:/$Recycle.Bin'):
        for file in files:
            if 'coords' in file.lower() and file.endswith('.pkl'):
                path = os.path.join(root, file)
                size = os.path.getsize(path)
                print(f"{path} | Size: {size}")

find_coords()
