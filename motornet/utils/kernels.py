import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import jet

class Kernel_Gaussian():
    def __init__(self,x_range, y_range, n_kernel, S, do_plot):

        self.x_range = x_range
        self.y_range = y_range
        self.n_kernel = n_kernel
        self.S = S * np.eye(2)
        self.do_plot = do_plot

        Kernel_x = np.linspace(min(self.x_range)+((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)),max(self.x_range)-((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)), self.n_kernel)
        Kernel_y = np.linspace(min(self.y_range)+((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)),max(self.y_range)-((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)), self.n_kernel)
        self.Kernel_x = Kernel_x
        self.Kernel_y = Kernel_y
        [m,n] = np.meshgrid(Kernel_x,Kernel_y)
        Kernel_mu = np.stack((m.flatten(),n.flatten()), axis=0).T

        self.centers = Kernel_mu

        [X,Y] = np.meshgrid(np.arange(min(self.x_range)-1,max(self.x_range)+1, 0.03), np.arange(min(self.y_range)-1, max(self.y_range)+1, 0.03))
        if do_plot:
            for K in range(len(Kernel_mu)):
                mu = Kernel_mu[K,:]
                Gaussian_pdf = lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
                Z = Gaussian_pdf(X.reshape(-1,1), Y.reshape(-1,1))
                Z = Z.reshape(X.shape)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap=jet)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Feature Value')
                plt.show()
                
            

    def encode(self,x,y):
        M = np.zeros((x.shape[0], len(self.centers)))
        for K in range(len(self.centers)):
            mu = self.centers[K,:]
            Gaussian_pdf =  lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
            M[:, K] = Gaussian_pdf(x,y)
        return M