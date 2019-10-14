Day 14:
----------
Milestones:
1. Read Capsule Networks from Cezanne Camacho's [blog post](https://cezannec.github.io/Capsule_Networks/). In a nutshell, capsule networks are another configurations of CNNs to detect hierarchically-related features as well as preserve properties/features related to these hierarchies such as width, orientation, color, etc.
2. Perused Andrej Karpathy's explanation of [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/#conv) specifically on how we can estimate parameters given the input, channels, expected number of kernels, stride and padding as well as the effect of pooling layers. This is a long read that I probably need to re-read this. 

I am currently implementing the CIFAR image classification exercise as added module of the CNN lesson and I am now in specifying model parameters. Will continue this tomorrow.

&#35;30DaysofUdacity


Day 13:
----------
Milestone:

Reviewed the Convolutional Neural Networks lesson. I had taken this lesson from Intro to Deep Learning with PyTorch free course but it is really nice to revisit the lesson again.

&#35;30DaysofUdacity


Day 12:
----------
Milestone:

Finished reviewing the sentiment analysis lesson and gearing up towards convolutional neural networks since I already finished the Deep Learning with PyTorch lesson. 

&#35;30DaysofUdacity


Day 11:
----------
Milestones:

1. Reviewed sentiment analysis lesson to reinforce the concepts of optimization and efficiency in training by dissecting the train and run functions. I probably need another day to finish reviewing the lesson. 
2. This is slightly related to deep learning that I needed sharing. I watched the [Full Frontal Rewind: The Best of Big Tech At Its Worst | Full Frontal on TBS](https://www.youtube.com/watch?v=C8AxAvh3-ck) with Samantha Bee. Bee jokingly featured deepfakes and facebook privacy fiascos since its inception and it made me realize the vast responsibility of AI enthusiasts and practitioners to prevent malicious use of AI and protect people's privacy. There is more to it so I would like you to check it out.

&#35;30DaysofUdacity


Day 10:
----------
Milestones: 

Finished Mini Projects 4, 5, and 6 of the Sentiment Analysis lesson.

Realizations: 

The lessons from the neural network stress the importance of understanding gradient descent and backpropagation in its fundamentality. Even Andrej Karpathy believes that is true. This lesson takes a notch to make me appreciate this truth even more. By creating a deep learning solution that is so verbose, you can observe patterns to convergence and speed and rectify them immediately with basic yet powerful tool - numpy. Although this touches the very basic neural network with easily computed gradients and errors, that is not the ultimate goal of this lesson. Its goal is to appreciate formulating solutions from the problems encountered from the results of initial attempts and build up from them, which I believe is the art of problem-solving in deep learning.

&#35;30DaysofUdacity


Day 9:
----------
Milestones:
1. Finished mini projects 2 and 3 of the sentiment analysis lesson. They are a bit tricky especially that I used comprehensions most of the time but it is a good practice for more comprehensions to come.

It has been two days since I started the sentiment analysis module and I appreciate the progression of forming an efficient solution to a particular problem. I am a bit slow but #30DaysofUdacity keeps motivating me. I hope you guys too.

&#35;30DaysofUdacity


Day 8:
----------
Milestones:
1. Finished Project 1 of the Sentiment Analysis lesson. It really is challenging when you are not looking at an already-made solution.
2. Learned that Counter package let's you count words easily and efficiently using few lines of code. I learned how to use Counter().update(), deviating from Trask's solution a little bit. I observed that it becomes tricky when you repeatedly update the Counter object since the update method does not exactly replace the previous value of each item in the dictionary, they are being augmented by the new values computed. But update method is proven useful for me in this case. 
3. I also refreshed my memory a little bit about dictionary comprehensions and map function. It really is become useful to make Trask's already few lines of code fewer. 

&#35;30DaysofUdacity


Day 7:
----------
This is my first week in the nanodegree and it has been a rewarding experience by far.
Milestones:
1. Finished, submitted and passed the first project Predicting Bike-Sharing Data. I got a training loss of 0.071 and validation of 0.155. I am now awaiting for code review.

![Day 7](./images/day_07.JPG)
2. Watched the history of [AI and Deep Learning](https://www.youtube.com/watch?v=ht6fLrar91U) visual podcast by Frank Chen where he explained how AI visions became fruitful on the present years from the summer of 1952, to AI winters, to breakthroughs in deep learning to the advances in Machine Learning products that we enjoy right now. I am glad that I watched this video. It feels like I am heading into the right direction of investing my time and efforts to learn this domain to produce systems enhanced with AI components. 

&#35;30DaysofUdacity


Day 6:
----------
Milestones:
1. Finally, I finished the Implementing Gradient Descent lesson. It is so much fun. This really reinforces the value of theoretical/mathematical intuition as before utilizing much more complicated deep learning methods. I probably need to review this again.
2. Watched Andrej Karpathy's [lecture](https://www.youtube.com/watch?v=59Hbtz7XgjM) about the importance of building the intuition behind backpropagation and gradient descent. 
3. Gone through the first project implementation (Predicting Bike-Sharing Patterns). Unfortunately, my validation error is 0.64. I probably need to review my implementation before submitting it. This is a challenging yet rewarding project at the same time.  

&#35;30DaysofUdacity


Day 5:
----------
Milestones:
1. I am halfway into finishing implementing gradient descent from scratch. I failed to finish it this time but I will surely finish them tomorrow.
2. I have been brushing up my skills in multivariable calculus and been watching several KhanAcademy videos in the process.

&#35;30DaysofUdacity


Day 4:
----------
Milestones:
1. Finished 67% of the Neural Network lesson. Most of the content that I had gone through were from the Intro to Deep Learning with PyTorch but I reviewed them anyway. It is really interesting how neural networks can be effectively learned through engaging visuals. I am loving it so far.
2. I had started learning how to implement gradient descent using NumPy and will finish the module tomorrow.

&#35;30DaysofUdacity


Day 3:
----------
Milestones:
1. Finished the whole first lesson: Introduction to Deep Learning.
2. I reviewed Jupyter Notebook and read the amazing and inspiring article about LIGO's journey in proving Einstein's theory a century ago - link [here](https://www.ligo.caltech.edu/news/ligo20160211). It is really incredible how open source Jupyter notebook that I used on several courseworks is also the tool that US research agencies used to communicate a massive feat. 
3. Went through the Matrix Math and Numpy Refresher. I received multiple lessons like that before. But Udacity's way of telling a story in an engaging and visual way helped me solidify the concept of matrix factorization even better. Loving every part of it.

&#35;30DaysofUdacity


Day 2:
----------
Milestones:
1. Finished the Anaconda Lecture of Lesson 1: Introduction to Deep Learning. I successfully updated my anaconda distribution and downloaded pertinent packages per coursework in the next Applying Deep Learning chapter.
2. Generated style transfers using my image as content image. 
![Day 2](./images/day_02.JPG)
3. Recently tried the Deep Traffic self driving car simulation. What I did was to adjust the lanesside to 2, patchesAhead to 3 while retaining patchesBehind to 0. I also created layer definitions, a total of 3 started at 256 number of neurons, to 128 and then 32. Ultimately, after training, I got a fair 64.54 mph on average. 
4. Explored a simulation of Reinforcement Learning applied autonomously in Flappy Bird game.

&#35;30DaysofUdacity


Day 1:
----------
Halfway into finishing the introduction to deep learning lesson of the Deep Learning Nanodegree.

I was late in starting the course. But as the old saying goes, it is better late than never. I am committed into finishing 30DaysofUdacity initiative from this day forward and I hope to dedicate at least 15 hours per week to my learning journey.

&#35;30DaysofUdacity
