import numpy as np
from PIL import Image
import h5py
import os

class ITrainable():
  pass
  def forward_propagation(self, X):
    raise NotImplementedError("Not Implemented")

  def backward_propagation(self, dY_hat):
    raise NotImplementedError("Not Implemented")

  def update_parameters(self):
    raise NotImplementedError("Not Implemented")

class DLLinearLayer(ITrainable):
  def __init__(self,name,num_units,input_size, alpha,optimize = None):
    self.name = name
    self.alpha = alpha
    self.optimization = optimize
    self.input_size = input_size
    self.num_units = num_units
    self.b = np.zeros((num_units,1),dtype = float)
    self.W_Xaviar_initialization()
    if(self.optimization == 'adaptive'):
      self.adaptive_cont = 1.1
      self.adaptive_switch = 0.5
      self.adaptive_W = np.full(self.W.shape,alpha,dtype = float)
      self.adaptive_b = np.full(self.b.shape,alpha,dtype = float)

  def set_W(self,W):
    self.W = np.copy(W)


  @staticmethod
  def normal_initialization(shape,factor=0.01):
    return np.random.randn(*shape)*factor

  def W_He_initialization(self):
    self.W = DLLinearLayer.normal_initialization((self.num_units,self.input_size,),np.sqrt(2/self.input_size))

  def W_Xaviar_initialization(self):
    self.W = DLLinearLayer.normal_initialization((self.num_units,self.input_size),1/np.sqrt(self.input_size))


  def save_parameters(self,file_path):
    file_name = file_path+"/"+self.name+".h5"
    with h5py.File(file_name, 'w') as hf:
      hf.create_dataset("W", data=self.W)
      hf.create_dataset("b", data=self.b)

  def restore_parameters(self,file_path):
    file_name = file_path+"/"+self.name+".h5"
    with h5py.File(file_name, 'r') as hf:
      self.W = hf['W'][:]
      self.b = hf['b'][:]

  def forward_propagation(self, prev_A):
    self.prev_A = np.copy(prev_A)
    Z = self.W @ prev_A + self.b
    return Z

  def get_W(self):
    return self.W

  def backward_propagation(self, dZ):
    self.db = np.sum(dZ, keepdims=True, axis=1)
    self.dW = dZ@self.prev_A.T
    return self.W.T@dZ


  def update_parameters(self):
    if self.optimization == 'adaptive':
      self.adaptive_W *= np.where(self.adaptive_W * self.dW > 0, self.adaptive_cont, -self.adaptive_switch)
      self.W -= self.adaptive_W
      self.adaptive_b *= np.where(self.adaptive_b * self.db > 0, self.adaptive_cont, -self.adaptive_switch)
      self.b -= self.adaptive_b
    else:
      self.W -= self.alpha * self.dW
      self.b -= self.alpha * self.db

  def __str__(self):
    s = self.name + " Function:\n"
    s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
    if self.optimization != None:
      s += "\toptimization: " + str(self.optimization) + "\n"
    if self.optimization == "adaptive":
      s += "\t\tadaptive parameters:\n"
      s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
      s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
    # parameters
    s += "\tParameters: W shape = "+str(self.W.shape)+", b = "+str(self.b.shape)+"\n"
    return s


class DLNetwork(ITrainable):
  def __init__(self,name):
    self.name = name
    self.layers = []

  def add_layer(self,iTrainable):
    self.layers.append(iTrainable)

  def forward_propagation(self, X):
    self.__X = X
    kelet = X
    for layer in self.layers:
      kelet = layer.forward_propagation(kelet)
    return kelet

  def backward_propagation(self, dY_hat):
    kelet= dY_hat
    for layer in self.layers[::-1]:
      kelet = layer.backward_propagation(kelet)
    return kelet

  def save_parameters(self,dir_path):
    path = dir_path+"/"+self.name
    if os.path.exists(path)== False:
      os.mkdir(path)
      for layer in self.layers:
        layer.save_parameters(path)

  def restore_parameters(self, dir_path):
    path = dir_path+"/"+self.name
    for layer in self.layers:
      layer.restore_parameters(path)

  def update_parameters(self):
    for layer in self.layers:
      layer.update_parameters()

  def __str__(self):
    s = self.name+'\n'
    for layer in self.layers:
      s+=layer.__str__()
    return s

class DLModel:
  def __init__(self,name,iTrainable,loss):
    self.name = name
    self.iTrainable = iTrainable
    self.loss = loss
    if(loss == 'categorical_cross_entropy'):
      self.loss_forward = self.categorical_cross_entropy
      self.loss_backward = self.dCategorical_cross_entropy
    if(loss =="square_dist"):
      self.loss_forward = self.square_dist
      self.loss_backward = self.dSquare_dist
    if(loss == "cross_entropy"):
      self.loss_forward = self.cross_entropy
      self.loss_backward = self.dCross_entropy

  @staticmethod
  def to_one_hot(num_categories, Y):
    m = Y.shape[0]
    Y = Y.reshape(1, m)
    Y_new = np.eye(num_categories)[Y.astype('int32')]
    Y_new = Y_new.T.reshape(num_categories, m)
    return Y_new

  def __str__(self):
    s = self.name + "\n"
    s += "\tLoss function: " + self.loss + "\n"
    s += "\t"+str(self.iTrainable) + "\n"
    return s

  def categorical_cross_entropy(self, Y_hat, Y):
    eps = 1e-10
    Y_hat = np.where(Y_hat==0,eps,Y_hat)
    Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
    errors = np.zeros(Y.shape[1])
    errors = -np.sum(Y*np.log(Y_hat),axis = 0)
    return errors

  def dCategorical_cross_entropy(self,Y_hat,Y):
    return (-Y+Y_hat)/Y.shape[1]

  def cross_entropy(self,Y_hat,Y):
    eps = 1e-10
    Y_hat = np.where(Y_hat==0,eps,Y_hat)
    Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
    return -(Y*np.log(Y_hat)+(np.full(Y.shape,1,dtype=float)-Y)*np.log(np.full(Y.shape,1,dtype=float)-Y_hat))

  def dCross_entropy(self,Y_hat,Y):
    eps = 1e-10
    Y_hat = np.where(Y_hat==0,eps,Y_hat)
    Y_hat = np.where(Y_hat == 1, 1-eps,Y_hat)
    return (1/Y.shape[1])*(-Y*Y_hat**(-1)+(np.full(Y.shape,1,dtype=float)-Y)*(np.full(Y.shape,1,dtype=float)-Y_hat)**-1)


  def square_dist(self, Y_hat, Y):
    errors = (Y_hat - Y)**2
    return errors

  def dSquare_dist(self, Y_hat, Y):
    m = Y.shape[1]
    dY_hat = 2*(Y_hat - Y)/m
    return dY_hat

  def compute_cost(self, Y_hat, Y):
    m = Y.shape[1]
    errors = self.loss_forward(Y_hat, Y)
    J = np.sum(errors)
    return J/m

  def confusion_matrix(self, X, Y):
    prediction = self.forward_propagation(X)
    prediction_index = np.argmax(prediction, axis=0)
    Y_index = np.argmax(Y, axis=0)
    right = np.sum(prediction_index == Y_index)
    print("accuracy: ",str(right/len(Y[0])))
    # print(confusion_matrix(prediction_index, Y_index))

  def backward_propagation(self,Y_hat,Y):
    dY_hat = self.loss_backward(Y_hat,Y)
    self.iTrainable.backward_propagation(dY_hat)

  def forward_propagation(self,X):
    return self.iTrainable.forward_propagation(X)

  def train(self,X,Y,num_iterations):
    print_ind = max(num_iterations // 100, 1)
    costs = []
    for i in range(num_iterations):
      Y_hat = self.iTrainable.forward_propagation(X)
      self.backward_propagation(Y_hat,Y)
      self.iTrainable.update_parameters()

      if i > 0 and i % print_ind == 0:
        J = self.compute_cost(Y_hat, Y)
        print("cost:",J,i/print_ind,"%")
        costs.append(J)
    costs.append(self.compute_cost(Y_hat, Y))
    return costs


class DLNeuronsLayer(DLNetwork):
  def __init__(self,name,num_units,input_size,activation,alpha,optimization=None):
    self.name = name
    self.linear = DLLinearLayer("Linear",num_units,input_size,alpha,optimization)
    self.activation = DLActivation(activation)
    super().__init__(name)
    super().add_layer(self.linear)
    super().add_layer(self.activation)


  def __str__(self):
    return self.linear.__str__()+self.activation.__str__()



class DLActivation(ITrainable):
  def __init__(self,activation):
    self.name = activation
    if activation == "tanh":
      self.forward_propagation =self.tanh
      self.backward_propagation =self.tanh_dZ
    elif activation == "relu":
      self.forward_propagation =self.relu
      self.backward_propagation =self.relu_dZ
    elif activation == "leaky_relu":
      self.leaky_relu_d = 0.01
      self.forward_propagation =self.leaky_relu
      self.backward_propagation = self.leaky_relu_dZ
    elif activation == "sigmoid":
      self.forward_propagation = self.sigmoid
      self.backward_propagation =self.sigmoid_dZ
    elif activation =='softmax':
      self.forward_propagation = self.softmax
      self.backward_propagation =self.softmax_dZ
    else:
      raise Exception("Undifiend activation")

  def sigmoid(self, Z):
    self.res = 1/(1+np.exp(-1*Z))
    return self.res

  def sigmoid_dZ(self, dA):
    self.dZ = dA*self.res*(np.full(self.res.shape,1,dtype=float)-self.res)
    return self.dZ

  def softmax(self,Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

  def softmax_dZ(self,dZ):
    return dZ

  def tanh(self, Z):
    self.res = np.tanh(Z)
    return self.res

  def tanh_dZ(self,dA):
    return dA*(1-self.res**2)

  def relu(self, Z):
    self.Z = Z
    return np.maximum(0,Z)

  def relu_dZ(self,dA):
    return np.where(self.Z <= 0, 0, 1)*dA

  def leaky_relu(self,Z):
    self.Z = Z
    return np.where(self.Z <= 0, self.leaky_relu_d*self.Z, self.Z)

  def leaky_relu_dZ(self,dA):
    return np.where(self.Z <= 0, self.leaky_relu_d, 1)*dA


  def update_parameters(self):
     pass

  def save_parameters(self,path):
    pass
  def restore_parameters(self,path):
    pass

  def __str__(self):
    return "Activation: "+self.name


class testGrad():
  def __init__(self):
    pass
  @staticmethod
  def check_grad(f,x,f_grad,epsilon = 1e-4,delta = 1e-7):
    aprox = (f(x+delta)-f(x-delta))/(2*delta)
    grad = f_grad(x)
    print(aprox,grad)
    diff = abs(aprox-grad)/(abs(aprox)+abs(grad))
    return (diff<epsilon,diff)

  @staticmethod
  def check_n_grad(f , parms_vec, grad_vec, epsilon=1e-4 , delta=1e-7):
    n = len(parms_vec)
    approx = np.zeros(parms_vec.shape)
    for i in range(n):
      pars_plus = np.copy(parms_vec)
      pars_plus[i]+=delta
      pars_min = np.copy(parms_vec)
      pars_min[i]-=delta
      approx[i] = (-f(pars_min)+f(pars_plus))/(2*delta)
    above = np.linalg.norm(approx-grad_vec)
    bottom = np.linalg.norm(approx)+np.linalg.norm(grad_vec)
    diff = above/bottom
    return (diff<epsilon,diff)


def get_digit_model():
    
    np.random.seed(1)

    Hidden = DLNeuronsLayer("Hidden",64,28*28,"sigmoid",0.1,'adaptive')
    Output = DLNeuronsLayer("Output",10,64,"softmax",0.1,'adaptive')

    digit_network = DLNetwork("digit_net")
    digit_network.add_layer(Hidden)
    digit_network.add_layer(Output)

    digit_model = DLModel("model",digit_network,'categorical_cross_entropy')
    digit_network.restore_parameters("parameters")
    
    return digit_model


def evaluateImage(img, digit_model):
     
    # Create a black background image
    if img.mode != 'RGBA':
      img = img.convert('RGBA')
      print(img.size)


    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Paste the inverted image onto the black background using the alpha channel as mask
    # black_bg.paste(inverted_img, (0, 0), inverted_img)
    white_bg.paste(img, (0, 0), img)

    # Convert to grayscale
    new_img = white_bg.convert('L')
    img_data = np.array(new_img).astype(np.float32) / 255 - 0.5

    img_data = img_data.reshape(784, 1)


    # Forward propagation
    Y_hat = digit_model.forward_propagation(img_data)

    percentageList = []
    for i in range(10):
      percentage = Y_hat[i][0] * 100
      percentageList.append([i, percentage])
    
    percentageList.sort(key=lambda x: x[1], reverse=True)
    
    # Get the predicted digit
    predicted_digit = np.argmax(Y_hat)
    percentageStr = '\n'.join([f"{i}: {percentage:.2f}%" for i, percentage in percentageList])

    return percentageStr