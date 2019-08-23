#Change the path to be the desired images --- image1 = base picture ----- image2 = style
image1_path = '/content/Myface.jpeg'
image2_path = '/content/Smileyface.jpeg'
#Open and identify the content image from its URL.
input_image = Image.open(image1_path)
#Set the height and width for the images
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
#Resize the content image passing in the width and height into the .resize() function
input_image = input_image.resize([IMAGE_HEIGHT,IMAGE_WIDTH])
#Save the newly sized image
input_image.save('MyFace.jpeg')
#Display the image
input_image
#Open and identify the style image from its URL.
style_image = Image.open(image2_path)
#Resize the input image passing in the width and height into the .resize() function
style_image = style_image.resize([IMAGE_HEIGHT,IMAGE_WIDTH])
#Save the newly sized image
style_image.save('HappyFace.jpeg')
#Display the image
style_image
#Data normalization and reshaping from RGB to BGR to convert images into a 
#suitable form for processing

#Convert the content image to an array using NumPys asarray() function
input_image_array = np.asarray(input_image, dtype='float32')

#Expand the shape of the array using NumPy's expand_dims(), so that we can 
#later concatenate the representations of these two images into a common data structure.
input_image_array = np.expand_dims(input_image_array, axis=0)

#Now we need to compress the input data by performaing two transformations
#1. Subtracting the RGB mean value from each pixel
#2. Changeing the ordering of array from RGB to BGR 
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]

input_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
input_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
input_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
input_image_array = input_image_array[:, :, :, ::-1]

style_image_array = np.asarray(style_image, dtype="float32")
style_image_array = np.expand_dims(style_image_array, axis=0)
style_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
style_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array = style_image_array[:, :, :, ::-1]

#Add the content image as a keras backend variable
input_image = backend.variable(input_image_array)

#Add the style image as a keras backend variable
style_image = backend.variable(style_image_array)


#Instantiate a placeholder tensor to store the combination image that retains 
#the content of the content image while incorporating the style of the style image.
combination_image = backend.placeholder(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

#Concatenate the context, style and combination image alongside the specified 
#axis 0 using keras backend.concatenate()
to_contatinate = [input_image, style_image, combination_image]
input_tensor = backend.concatenate(to_contatinate,axis=0)

#Using the pre-trained VGG16 model(16 layer model)- one of Keras applications. 
#It is a convolutional neural network tained on ImageNet include top is whether 
#to include the 3 fully-connected layers at the top of the network
#since we are not interested in image classification we set this value to false
model = VGG16(input_tensor=input_tensor, include_top=False)
#Construct a dictionary to easily look up layers by their names
layers = dict([(layer.name, layer.output) for layer in model.layers])
CONTENT_WEIGHT = 0.02

#For the content loss, we draw the content feature from the block2_conv2 layer.
content_layer = "block2_conv2"
layer_features = layers[content_layer]

content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

#The content loss is the squared Euclidean distance between content and combination images.
#Use backend sum() and square() functions
#CONTENT_LOSS = backend.sum(backend.square(combination_features)+backend.square(content_image_features))
CONTENT_LOSS = backend.sum(backend.square(combination_features - content_image_features))
print(CONTENT_LOSS)
loss = backend.variable(0.)
loss += CONTENT_WEIGHT * CONTENT_LOSS

def gram_matrix(x):
    #Turn a nD tensor into a 2D tensor with same 0th dimension. In other words, 
    #it flattens each data samples of a batch.
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    
    # .dot Multiplies 2 tensors (and/or variables) and returns a tensor.
    gram = backend.dot(features, backend.transpose(features))
    return gram

CHANNELS = 3
STYLE_WEIGHT = 4.5

def compute_style_loss(style, combination):
    style = gram_matrix(style_image)
    combination = gram_matrix(combination_image)
    size = [IMAGE_HEIGHT,IMAGE_WIDTH]
    style_loss = backend.sum(backend.square(style_image - combination_image))
    return style_loss / (4. * (CHANNELS ** 2) * (size ** 2))
  
#The style layers that we are interested in
style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]

for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss = backend.sum(backend.square(style_image - combination_image))
    loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]

# Now we have our total loss , its time to optimize the resultant image.We start by defining gradients
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:

    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

ITERATIONS = 5

#This resultant image is initially a random collection of pixels, so we use 
#(fmin_l_bfgs_b - Limited-memory BFGS) which is an optimization algorithm
for i in range(ITERATIONS):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, 
                                  x.flatten(), 
                                  fprime=evaluator.gradients, 
                                  maxfun=20)
    print("Iteration %d completed with loss %d" % (i, loss))
    
#To get back output image do the following
x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
x = x[:, :, ::-1]
x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
x = np.clip(x, 0, 255).astype("uint8")
output_image = Image.fromarray(x)
output_image.save("output.png")
output_image

#Save a combined image to file
combined = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
x_offset = 0
for image in map(Image.open, ['MyFace.jpeg', 'HappyFace.jpeg', 'output.png']):
    combined.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
combined.save('FacecombinedImage.png')




