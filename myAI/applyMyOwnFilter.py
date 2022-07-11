import numpy as np

# filter = [[0,-2, 0], [-1, 2, -1], [0, 2, 0]]
# filter = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

def ApplyFillter(i:int , data: list, kernelSize: tuple[int, int], fillter: list[list], photoSizeX:int, photoSizeY:int , stride: int  = 1):
    PhotoAfterFillter = []
    fillterMatrix = np.array(fillter)
    pixelTimes = 0
    rowTimes = 0
    data = np.array(data)
    photo = data.reshape(photoSizeX, photoSizeY)  
    for n in range(int(np.ceil((photoSizeY - kernelSize[1]) / stride))):     
        for i in range(int(np.ceil((photoSizeX - kernelSize[0]) / stride))):
            kernel = []    
            for row in range(stride * rowTimes, (kernelSize[1] + (stride * rowTimes))):
                pixelList = []
                for pixel in range(stride * pixelTimes, (kernelSize[0] + (stride * pixelTimes))):
                    pixel = photo[row][pixel]
                    pixelList.append(pixel)
                kernel.append(pixelList)
            kernel = np.array(kernel)
            newPixel = kernel * fillterMatrix
            PhotoAfterFillter.append(newPixel.sum())
            pixelTimes += 1
        rowTimes += 1
        pixelTimes = 0
    PhotoAfterFillter = np.array(PhotoAfterFillter)
    PhotoAfterFillter = PhotoAfterFillter.reshape(int(np.ceil((photoSizeX - kernelSize[0]) / stride)), int(np.ceil((photoSizeY - kernelSize[1]) / stride)))
    return PhotoAfterFillter

