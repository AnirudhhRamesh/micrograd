import streamlit as st
from micrograd.schemas.schemas import ModelSchema
from micrograd.nets.NeuralNet import NeuralNet
from micrograd.engine.value import Value
from PIL import Image
import numpy as np
import json

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    weights_file = "model_weights_0-79.json"
    model = load_model(weights_file)

    # Prepare the streamlit interface
    st.title("The World's Worst Hotdog Classifier")
    st.write("Hey! My name is Arnie. I'm a MSc CS student @ ETH Zurich. 👋 Here's my [Twitter](https://x.com/arnie_hacker), [LinkedIn](https://www.linkedin.com/in/anirudhhramesh/) and [portfolio](http://anirudhhramesh.github.io)")
    st.subheader("What is this?")
    st.write("Inspired by Jian-Yang from the show 'Silicon Valley', I decided to build a hotdog classifier.")
    st.write("My classifier has a training loss of 0.8 and a training accuracy of 52%. Making it quite possibly the world's worst hotdog classifier :)")

    st.image("jian_yang_hotdog.png", caption="Jian-Yang's hotdog classifier, from the show Silicon Valley", width=600)
    
    st.subheader("So why is this cool anyway?")
    st.write("""
    1. I built the classifier *entirely* from scratch. No existing models, no ML libraries, no step-by-step tutorials.
    2. I did not follow any tutorials while building it. I had watched Andrej Karpathy's Micrograd video in May, and then finished the DeepLearning.ai ML Specialization. I built this during July though, mainly through first principles reasoning and recalling fundamentals from DeepLearning.ai, deriving the backpropagation algorithm from scratch, thinking through e.g. sigmoid function, Value-Neuron-Layer system design, etc
    3. I learnt ML & built this project in evenings & weekends, while juggling a full-time software internship at Amazon Web Services (and doing sports 4x per week + coding other random projects)!
    4. Most importantly - this is only the start of my grind :)
    """)
    
    st.divider()

    st.subheader("Test out the model yourself! 👇")
    
    # Display images 1-4 in the same row
    col1, col2 = st.columns(2)
    with col1:
        st.image("hotdog1.jpg", width=200, caption="hotdog1.jpg")
        st.image("hotdog2.jpg", width=200, caption="hotdog2.jpg")
    with col2:
        st.image("not_hotdog1.jpg", width=200, caption="not_hotdog1.jpg")
        st.image("not_hotdog2.jpg", width=200, caption="not_hotdog2.jpg")

    # Select between 4 images
    selected_image = st.selectbox("Select an image", options=["hotdog1.jpg", "hotdog2.jpg", "not_hotdog1.jpg", "not_hotdog2.jpg"])

    uploaded_file = st.file_uploader("Or, upload your own image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        selected_image = uploaded_file

    # Display the selected image
    st.image(selected_image, width=200)

    # A button which predicts the image
    if st.button("Predict"):
        st.write(f"Predicting {selected_image}...")
        st.write(f"{predict(selected_image, model)}")
    
    st.divider()
    st.title("Some notes!")

    st.write("""
    Finishing the ML specialization & building this fun project (and more random coding & sports) basically led to me working ~10-12hrs per day for the past 2 months :) 

    But I'm not at all burnt out! I'm so happy to be able to finally learn and get through all these concepts through pure curiousity. The past semesters I had spent building a start-up (even interviewing for YCombinator) but:
    """)

    st.markdown("""
    - In my YC interview question, they asked the simple question 'why do you want to build [this](https://app.swirl.so)?'. And I couldn't answer it. I realized I hadn't figured out what purposeful thing I wanted to work on for the next 10+ years
    - After getting on Twitter around March 2024, I had a growing bucket list of cool projects I wanted to build. Machine Learning, Robotics, LLMs, .... It felt so amazing to finally have time to work on these projects (with no deadlines or limitations!)
    """)

    st.write("I'll be sharing more projects I'm building over the Summer (and beyond!) - so go follow me on [Twitter](https://x.com/arnie_hacker) (where I document my daily grind) and [LinkedIn](https://www.linkedin.com/in/anirudhhramesh/)!")

    st.divider()

    st.header("Future improvements")

    st.markdown("""
    1. Add numpy support or write my own CUDA methods to support GPU-accelerated calculations. This would make training much faster and support larger neural networks.

    2. Add convolutional neural networks (CNN) support. CNNs work by reading multiple smaller rectangles of the image. This allows it to focus on finer details (and I believe makes it somehow equivalent to data augmentation). This should drastically improve the performance.
    """)

    st.write("""
    The above are two incredibly simple improvements that would lead to huge performance improvements. However, I decided to skip implementing these for now because I accomplished what I set out to do (validating my ML understanding by implementing backprop from scratch). I may come back one day to clean everything up and implement these though :), although since I have a fairly-good understanding already it makes it less valuable to implement for such as 'sandbox-learning' project.
    """)

    st.header("References & resources used")

    st.write("""
    The two main resources that helped out building Micrograd was Andrej Karpathy's [Micrograd Lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) and DeepLearning.ai [ML Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
    As explained above, I built this project with no tutorials or step-by-step guide, but trying my best through first-principles reasoning and from the ground up. Hopefully my technical deep dive below shows and demonstrates my reasoning.

    That being said, I did watch the Micrograd lecture once once, in May, which is why design-wise and intuition-wise the project is similar (Value, Neuron, Layer). I started coding the project in ~July, without rewatching the lecture in order to properly test my understanding and go through the reasoning steps myself, since I had mostly forgotten all the actual implementation details.
    Most of the intuition behind building this was from the ML Specialization, but also some calc stuff I did from high-school/uni.

    On a further note, once I had a working micrograd, I faced some problems with getting increasing loss during the training steps. After trying to debug a couple evenings after work, manually computing the backprop and not figuring out why it was increasing, I decided to look through Karpathy's [codebase](https://github.com/karpathy/micrograd) to debug faster (as my primary objective for the project was to solidify my ML understanding which I accomplished by building a working version).
    """)

    st.markdown("""
    - I used this skim-through to improve my code quality for the Neuron and Layer methods (I had some long loops where karpathy's was elegant one-liners). This fitted my secondary objective of improving my coding & code quality
    - I also copied over some Value functions which I simply did not know existed (such as __rneg__ and __rmul__), just in case + it made the implementation cleaner
    - I eventually found out the bug was mostly because I forgot to zero_grad which meant my gradients were stacking up between steps (and thus the loss function was increasing) 💀
    """)

    st.divider()

    st.title("Technical Deep Dive")

    st.write("""
    Note: This is a very scrappily put-together 'train-of-thought' explanation of my understanding of ML and how I derived the micrograd/backprop from the ground up.
    It's mostly useful to myself to re-iterate once again all the components of ML and also to prove to others my understanding of ML. I wrote this up in a couple hours from purely memory.
    I'll put more time into a cleaner explanation of this if you ping me on LinkedIn/Twitter, otherwise there's a bunch of far better explanations of ML than can be found e.g. through deeplearning.ai's ML Specialization or @Andrej Karpathy's Micrograd lecture.
    """)

    st.header("Some background knowledge on what is machine learning")

    st.write("""
    My understanding of machine learning is that it's just maths. It basically comes down to fitting a line to a set of points. The closer the line is fit to these points, then the better the predictions will be (when predicting something, we basically sample the function at the specified point).
    """)

    st.subheader("So how do we fit a line?")

    st.write("""
    In high school, we solved closed-form solutions by hand (i.e. where an absolute solution exists). This works fine for two or three points. 

    For a very simple example, we basically try to fit a line y = m * x + b, where we must solve for m and b. m in this case is the gradient, for 2 points can be easily found as the difference of y over the difference in x.

    But as the complexity increases, it becomes very hard to find a closed form solution. E.g. how do you find each m when we have more features like y = m1*x1 + m2*x1 + m3*x1 + b?

    This is why we try to approximately fit the line.
    To do this, we need to have a measure of 'how well does our line fit the points', so that we can continually try to bring this measure to zero ('optimize this').
    """)

    st.subheader("Defining 'how well our line fits the points'")

    st.write("""
    We can basically use the metric of 'how far apart are the points from my line?'.  

    This is straightforward: 
    1. For a given x, we sample our line/function at x and subtract the actual value (y) from it, and this gives the difference. 
    2. We'll have to square this value as we want it to be 'absolute' distance. . If we don't, then when summing up (in 3.), we could inadvertently cancel out differences if e.g. one is positive and one is negative.
    3. We'll sum these square differences over all the points we have to get the total loss
    4. We take the mean over the total data points (so that adding more data samples does not drastically change our metric). 

    This basically describes the loss, in this case using the Mean Squared Error (MSE) function and my understanding of it. And this will be our measure of 'how far apart are my points'.

    Here is the MSE loss function:
    MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
    """)

    st.subheader("Fitting our line to our data points")

    st.write("""
    Now that we have defined our MSE loss function, we want to bring it down to zero (which would mean the distance between the points and our function is 0, i.e. our line fits our points and thus can accurately predict y for new points).
    """)

    st.subheader("How can we optimize this function?")

    st.write("""
    From the way we defined it, the loss is a function of the coefficients of x (in this case, m and b).
    By varying m and b, we change our function which changes the prediction and thus the loss.
    So, we want to find out which direction to vary m and b so that the difference between the prediction and the actual y gets smaller.
    """)

    st.subheader("How can we figure out which direction to vary m and b?")

    st.write("""
    A naive solution would be to compute the loss for every m (and likewise for b) (equivalent to plotting e.g. loss against m)
    We then pick the loss which results in the lowest for m and b.
    - This would be incredibly expensive to compute the loss for every possible value of m and b (as these are continuous values).
    """)

    st.image("loss_against_m.png", caption="Loss against parameter m")

    st.write("""
    A more efficient way (which avoids computing every possible value of the parameters), would be to look at the change in the loss with respect to m and b.

    As a reminder, we are trying to find the m which results in the lowest loss.
    Let's say currently we are at m=2, the loss is approximately 10.
    By taking the partial derivative of the loss with respect to m, we are basically asking 'how does the loss change when m changes.
    For a given point, we can use the formula for the limit/derivative:
    f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}

    This tells us that if we take a very small h e.g. 0.001, and we compute the loss w.r.t to m=2, we get that the (f(2.001) - f(2)) / 0.001 = (9.999 - 10) / 0.001 = (-0.001) / 0.001 = -1 (very rough calc)

    What does this -1 means? It means that if we increase m by 1, then the loss will increase by -1.

    This is indeed the direction we want to go in (we want the loss to decrease), thus we can take a step in this direction.

    So m = m - dL/dm (subtract by a negative value, thus we are adding, so m increases which means loss decreases)

    Now if we do the same thing for m=8, we see the dL/dm is approximately 1. So it means if m increases by 1, then the loss increases by 1. We want to reduce the loss, thus we want to decrease m (which decreases loss). Conveniently, our previous function handles this correctly.

    So m = m - dL/dm (subtract by a positive value, thus m decreases thus loss decreases)

    This formula is our 'optimization step'. We can improve it further by adding a hyperparameter epsilon which tells us how big optimization steps to make (if we take too big steps, we could jump across the minima. If we take too small steps, we could converge at a local minima or take too long to train):
    m := m - \\epsilon \\frac{\\partial L}{\\partial m}

    Eventually, we'll get to a point where the loss barely changes when we change m, which is the minimum of the loss (it is where the loss is changing from positive to negative, and thus the change in the loss in the lowest (and equal to 0)). This would be where our loss is at its lowest.
    """)

    st.image("loss.gif", caption="Loss visualized")

    st.write("""
    If we take a 2nd derivative of the loss w.r.t to w, this would be a straight line which is at y=0 when the change in loss is zero, and thus parameter m is 'optimal'.

    This basically is how gradient descent works and the foundation on top of which ML is built on.

    We can expand this to further dimensions easily, as it's just a matter of taking the partial derivatives w.r.t. all the parameters/dimensions we have.
    """)

    st.header("Logistic regression (classifier)")

    st.write("""
    For classification (e.g. classify an image as hot-dog or not), we use the logistic regression function, as we want to predict discrete values such as 1, 0 rather than continuous values that linear regression predicts such as 1.3232 or -12.42

    Logistic regression works in mostly the same way as linear regression. Except that, with logistic regression we'll want to define the binary cross entropy loss instead of the MSE loss.

    We construct this in a way that our loss increases when our prediction is incorrect, and our loss does not change when our prediction is indeed correct. Eventually, if all our predictions are correct, then our loss would not change thus it will be zero.

    The binary cross entropy loss is basically the following:
    binary cross entropy cost = -y * log (y^) -(1-y) * log ( 1-y^)

    Just like the MSE, I love this function because it's actually very intuitive: 
    - The log function tells us what exponent we need to raise our base to get the value. So e.g. log2(8) = 3, because we need to raise 2 (base) to 3 (exponent) to get 8 (value). 2^3 = 8.
    - The log of 1 is equal to zero (e.g. log2(1) = 0). So for correct predictions, we want the value inside the log to be as close as possible to 1.
    - The log of (close to) 0 approaches minus infinity (e.g. log2(0.5)=-1, log2(0.25) = -2). For an incorrect prediction, we want the value inside the log to be close to zero (so that the loss increases).

    Note: As you can see for incorrect guesses (i.e. less than 1), the log function outputs a negative value), thus we need to negate the logs in our function (which is why it's -log).
    Note: We multiply the logs by y and (1-y) because this serves as a 'switch'. If y = 1, we ignore the loss for y=0 (because 1 - y=1 = 0, which cancels the second part of the equation).

    Note: As you might have noticed, since we're using logs, we need to ensure that our predictions are capped between 0 and 1. log(0) or negative does not exist. In order to do this (and also to better fit our data points), we'll have to use a sigmoid cost function (instead of the simple linear cost function of y= m*x + c).

    The sigmoid function as as follows:

    y^ = 1 / 1 + (e^-(mx + b))

    This is also intuitive:
    - When the exponent component is negative, then our prediction is 1/(1 + huge) = 0
    - When the exponent component is 0, then our prediction is 1/(1+1) = 0.5
    - When the exponent component is positive, then our prediction is 1/(1 + small) = 1
    """)

    st.image("sigmoid.png", caption="Sigmoid")

    st.markdown("===")

    st.header("Neural nets")

    st.write("""
    Neural nets basically build upon chaining multiple linear regression/logistic regression type functions together, except taking architecture-inspiration from the human neuron.

    The human neuron consists of a 'neuron' with multiple 'dendrites' that activate the neuron's output.

    Likewise, for a neural network neuron, we take in multiple signals (in this case, it would be the weights * input x), and output an output which is a sum of the weights.
    """)

    st.image("neuron.png", caption="Neuron")

    st.write("""
    By chaining together multiple 'neurons', we able to make more complex functions that can better fit arbitrary data points (which a simple line function would not be able to capture).
    """)

    st.image("complex_data.png", caption="Complex data")

    st.write("""
    Adding more layers and weights generally does not overfit our data (our models tend to perform better) simple because of the regularization applied between layers(?) Or weights/links can be set to zero whenever need(?)
    """)

    st.header("Autograd")

    st.write("""
    So the beauty of autograd/auto-diff (automatic differentiation) is that it efficiently computes the partial derivatives w.r.t. the loss.

    Computing the partial derivative of each parameter with respect to the loss would be O(n^2) complexity. With autograd, we are able to do this in one-pass in O(n) complexity.

    The naive, inefficient way would be to compute the partial derivative with respect to the loss for each parameter, which would be O(n^2), where n is the # of parameters.

    The auto-grad, more efficient way would be do a so-called 'forward-pass' (inference) which passes forwards the values, then a 'backwards-pass' which pass back the partial derivatives.

    In the forward pass, we very simply chain together multiplications that lead to the function we have defined (e.g. y^ = m * x + b)
    In the backward pass, we recursively applying the chain rule from differentiation.

    Refresh on the chain rule:
    dy/dx = du/dx * dy/u

    What the chain rule states is that the change in loss with respect to x is equal to the change in loss with respect to an intermediate variable u * the change in u with respect to x.

    This means we can go back from the loss, and at each previous node we can apply the chain rule to get the partial derivative of that node with respect to the loss. As a result, computing the partial derivatives now only takes O(n)!

    Once the back prop is complete, all our parameters now have their partial derivatives with respect to the loss.

    Thus we can apply the optimization step (which is basically updating the parameters by subtracting the partial derivatives). 
    """)

    st.header("Explaining the code")

    st.subheader("Micrograd / Engine / Value.py")

    st.write("""
    With micrograd, we model the autograd using the micrograd / engine / Value.py
    Here, we basically model each data sample/parameter as a node in our computation graph (which stores a value and a grad=(partial derivative w.r.t the loss)

    When we do an addition (e.g. c = a + b), y is the child of x and b. When we do backprop, from the c node we can compute the partial derivatives of a and b using chain rule. From this, a would be equal to the derivative of loss w.r.t to c, b would also be equal to the derivative of loss w.r.t to c. This is because in c = a + b, when icreasing a by 1, c will also increase by 1. 
    When we do a multiplication (e.g. c = a * b), c is the child of an and b. When we do backprop, from the c node we can compute the partial derivatives of a and b using chain rule. From this, a would be equal to b * partial derivative of c. b would be equal to a * partial derivative of c. This is because in c = a * b, when we take the derivative, we are left with dc/da = b. Thus when a increases by 1, c increases by 1*b times.

    We then add support to several other operations such division, exponent (needed for sigmoid function), power (needed for MSE loss). I also added in some __rneg__ or __rmul__ methods etc, after viewing Andrej Karpathy's code as I handn't thought about these/been aware of these and was debugging (wondering whether these might solve some of my incorrect gradient descent problems).
    """)

    st.subheader("Micrograd / Engine / Neuron.py")

    st.image("neuron.png", caption="Neuron")

    st.write("""
    Here, the neuron models how the brain neuron works. This diagram shows what a neuron looks for multiple regression (i.e x has multiple features such as x1 = (1.2, 2.0) ). For a simple linear regression with one feature of x (e.g. xi = (1.3), then we will only have w and b as parameters, and all samples will be multiplied by the same w.

    The neuron's output is then very simply the sum of the weights * x and then added to b (so y = m*x + b).

    Note: I wrote some for loops for this that worked, but while debugging, took the opportunity to re-use karpathy's cleaner 2-liner functions using zip. Again, I achieved what I set-out to do (which is solidifying my ML understanding) which is why I decided to improve my code by referencing an expert programmer (to achieve my secondary goal of writing cleaner code).
    """)

    st.subheader("Micrograd / Engine / Layer.py")

    st.write("""
    Neural nets generally have 'layers'. These are basically just a line of neurons together. All the inputs of the neurons will be the same (from the inputs or the previous layer's outputs). The outputs will be the layer's neurons' outputs, and propagate to the next layer.
    The layer abstraction allow us to set an activation layer to every neuron of this layer. We also have layers so that we can apply multiple neurons to the same input to extract different features out of it.
    """)

    st.subheader("Micrograd / Engine / Model.py")

    st.write("""
    The final layer abstraction encapsulates all layers into a single model. This allows us to easily define and connect all the layers' inputs/outputs together. From here we can make predictions/inferences for a given sample x and get the output.
    """)

    st.subheader("Loss function")

    st.write("""
    The loss function takes the output of the model and calculates the loss. From this we can run backdrop (loss.backprop()), and then update the weights using an optimizer. The two losses currently are MSE (for linear regression) and Binary Cross Entropy (for simple classification).
    """)

    st.header("Training")

    st.image("model_loss.png", caption="Training loss")

    st.write("""
    Since my learning objective was to solidify my ML foundational understanding and ship a hotdog classifier, I did not spend too much thought towards getting an amazing model.

    For the training loss, I got around 0.85 after many runs. My learning rate was definitely too high (which probably explains why my loss descent increases rather than slows down). However, when using a smaller learning rate, the training simply took too long and kept getting stuck in local minima (around 1.2-1.3).

    To resolve this, I attempted a very naive solution of adjusted the learning rate at different thresholds (so decreasing the learning rate as it got closer, because the steps were increasing). This still did not work, as you can see the loss was bouncing (and if I picked something very small it would again get stuck in local minima).

    I tried adding stochastic gradient descent (which only updates a random subset of the weights, and supposedly reduces the chance of getting stuck in local minima), but I ended up scrapping this because it increased the learning rate.

    Using larger neural nets would fit the data better, however slowed down training too much (as I hadn't invested in NumPy primitives during the construction of the project or CUDA-optimizations - maybe when I have free time in the future though!). I also didn't spend much time on convolution neural nets (because again, I achieved my primary learning objective and was trying to ship fast).
    """)

    st.header("Connecting the dots")

    st.write("""
    Doing these computations becomes incredibly slow on a CPU. We're talking x calculations per sample and weight. And all these computations are very homogeneous (a bunch of multiplications) and additions. This is where linear algebra and GPUs come in.
    We can simple map these calculations in matrix transformations. Once we have matrices, these are basically applying the same operation to several cells of the matrices - something that GPUs are great at!

    With GPUs, we can use e.g. CUDA from Nvidia to apply a specific function to a huge amount of values in parallel. GPUs are support very simple functions but incredibly parallel, whereas CPUs are incredibly complex and generally single-computation or very low parallelism.
    """)

    st.subheader("Note on optimizations")

    st.write("""
    There are additional types of optimizations steps, such as stochastic gradient descent, where we only optimize some weights. This helps us to avoid local minimum.
    Another technique is adding momentum - as we get closer and closer to the minimum, the change in loss will decrease thus the optimization steps will be smaller and smaller. Thus we can add momentum (= the average of the previous gradients) which keeps our steps high and prevents local minima)
    """)

    st.subheader("More things")

    st.write("""
    Regularization - regularization prevents overfitting by ensuring we try to minimize the weights and thus focusing only on the most important features (rather than overfitting to every feature). Intuitively, this allows to automatically select the features which contribute the most to the output and reduce those that do not. This is also known as weight decay.
    This is very simple - just add the sum of the weights to the optimization function (e.g. MSE), and multiply it by a hyperparameter lambda. The optimization function will seek to minimize the loss, thus will reduce the weights themselves to zero (which is inherently feature selection). A high lambda would prioritize less overfitting (as we want more weights to be zero).
    """)
    
def load_model(file:str):
    # Parse weights from weights.json
    logger.info(f"Loading model weights from {file}...")
    with open(file, 'r') as f:
        model_json = f.read()
    
    validated_model = ModelSchema.model_validate_json(model_json)
    
    model = NeuralNet.from_json(validated_model)
    logger.info(f"Model loaded from {file}")
    
    return model

def predict(image:str, model):
    prediction = model(convert(image))

    predicted = "Hotdog" if prediction.value > 0.5 else "Not Hotdog"
    return f"Prediction: {predicted}, Confidence: {prediction.value:.2f}"

def convert(image_path, image_size=225):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image = image.convert('L')  # Convert to grayscale
    img_array = np.array(image).flatten() / 255.0  # Normalize to [0, 1]
    return img_array

def render():
    st.write("Hello World")

if __name__ == "__main__":
    main()