import matplotlib.pyplot as plt


class ImagePlotter:
    def __init__(self, image_number=1):
        self.x = []
        self.y = []
        self.image_number = image_number
        for _ in range(image_number):
            self.x.append([])
            self.y.append([])

    def append(self, input_x, input_y, index=0):
        self.x[index].append(input_x)
        self.y[index].append(input_y)

    def animation_plot(self):
        for i in range(self.image_number):
            plt.plot(self.x[i], self.y[i],label = f"index{i}")
        plt.legend()
        plt.show(block=False)
        plt.pause(0.05)

        plt.clf()

    def static_plot(self, index=0):
        plt.plot(self.x[index], self.y[index])
        plt.show(block=True)
        plt.clf()
    def reset(self):
        self.x = []
        self.y = []
        for _ in range(self.image_number):
            self.x.append([])
            self.y.append([])

# import numpy as np
# x = np.linspace(0,100,100)
# y = np.sin(x)
# img = ImagePlotter()
# for i in range(len(x)):
#     img.append(x[i],y[i])
#     img.animation_plot()
