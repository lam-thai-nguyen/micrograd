micrograd

--Overview
This project - micrograd - implements backpropagation and the training of a neural network from scratch.
The author decides to make this after watching Andrej Karpathy's Lesson 1 of the course Neural Networks: Zero To Hero.
The purpose of this project: challenging the understanding of the author.

***

If you can't explain it simply, you don't understand it well enough. - Albert Einstein

***

--Introduction
In the field of machine learning, students (the author!) often overlook basic concepts like backpropagation or optimization and go straight to YOLO (object detection) or GPT (chatbot).
However, going back to the basics teaches us a lot.
---> This project implements the training of a neural network from scratch (view the Scope section). It is based on but is not a copy of Andrej Karpathy's lesson 1 of the course Neural Networks: Zero To Hero. 

--Problem statement
Given a set vectors and desried binary outputs, fit a neural network to the data.
Input # desired output
[
    [1, 2, 3], #1
    [2, 3, 4]. #0
    [3, 4, 5], #1
    [4, 5, 6], #0
    [5, 6, 7], #1
]

--Scope
To achieve our goals, there are a few things we need to implement
1) backpropagation
2) An Multi Layer Perceptron (MLP)
3) The training pipeline

--What are the contributions of this project?
Nothing really. The author makes this to challenge his understanding of the topic.

--Did the author just copy Karpathy's code?
No. The author's version is worse. Because his understanding is nowhere near that of Karpathy.
But that is the point. The author wants to test his understanding by re-implementing everything by himself.
However, the author might adopt some ideas or structure his project like that of Karpathy.

--How should I view this project?
View this project in this order:
demo.ipynb: demonstrates the training of a neural network for the moon dataset. This is normally performed using scikitlearn. But here we implement everything. No numpy.
test/test_engine.py: shows 2 tests that compare the implementation with PyTorch. This gives an intuition about what is implemented.
To run the test: pip install pytest ---> create a blank conftest.py at the top level ---> in powershell terminal: pytest test/test_engine.py 
micrograd/ and micrograd.ipynb in parallel: micrograd/ is the final result, whereas the notebook shows the thinking process.

--Reference
Karpathy lesson: https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

--Further read
The author's first note while watching Karpathy's lesson: https://drive.google.com/file/d/1VnLI6hKJkT7K67WLTeW-XBMBbcuDuRWo/view?usp=sharing
